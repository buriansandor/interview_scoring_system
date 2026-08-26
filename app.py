import os
import json
import logging
import difflib
from pathlib import Path
import pandas as pd
from google import genai
from google.genai import types
import security, guardrails

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler("agent.log", encoding="utf-8")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

startup_cwd = Path.cwd().resolve()
logger.info("[INFO]\tStartup working directory: %s", startup_cwd)

def try_load_api_key_from_key_file() -> str:
    """
    Find the .key file and set the GEMINI API key based on the content of the file.
    """
    key_files = sorted(Path(".").glob("*.key"))
    if not key_files:
        return ""

    for key_file in key_files:
        try:
            key_value = key_file.read_text(encoding="utf-8").strip()
            if key_value:
                os.environ["GEMINI_API_KEY"] = key_value
                logger.info("[INFO]\tLoaded GEMINI_API_KEY from %s", key_file.name)
                return key_value
            logger.warning("[WARNING]\t%s is empty.", key_file.name)
        except OSError as e:
            logger.warning("[WARNING]\tFailed to read %s: %s", key_file.name, e)
    return ""

def load_the_JSON():
    """
    This function return the JSON object from the JSON file which is in this folder.
    """
    json_rule_files = sorted(Path(".").glob("*.json"))
    if not json_rule_files:
        logger.error("[ERROR]\tNo JSON rule files found.")
        return None
    if len(json_rule_files) > 1:
        logger.warning("[WARNING]\tMultiple JSON files found. Using the first one: %s", json_rule_files[0].name)
    with open(json_rule_files[0], "r", encoding="utf-8") as f:
        return json.load(f)

def load_the_CSV():
    """
    This function returns a data frame from the CSV file which is located in this folder.
    """
    csv_files = sorted(Path(".").glob("*.csv"))
    if not csv_files:
        logger.error("[ERROR]\tNo CSV files found.")
        return None
    if len(csv_files) > 1:
        logger.warning("[WARNING]\tMultiple CSV files found. Using the first one: %s", csv_files[0].name)
    return pd.read_csv(csv_files[0])

############# Agent ########################
# 1. Definiáljuk az eszközöket (Tools), amiket az ágens használhat
def range_and_type_checker():
    """
    If the agent provides a value outside the scale, this tool immediately forces the model to 
    regenerate the output with an error message 'The score is not within the specified range; please try again.' 
    before the data is written to the output file.
    """


def confidentality_enhance():
    """
    If the model is uncertain (confidence below 0.5), 
    we do not accept the low-quality prediction; 
    instead, we initiate a guided re-evaluation (Self-Refine) loop.
    """


# Biztonságos eszköztár regisztráció
my_tools = [range_and_type_checker, confidentality_enhance]
tool_config = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="range_and_type_checker",
            description="If the agent provides a value outside the scale, this tool immediately forces the model to regenerate the output.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"],
            },
        ),
        types.FunctionDeclaration(
            name="confidentality_enhance",
            description="If the model is uncertain, initiate a guided Self-Refine loop.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"],
            },
        ),
    ]
)

api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key:
    api_key = try_load_api_key_from_key_file()

if not api_key:
    logger.warning('[WARNING]\tGEMINI_API_KEY is not set. Stopping program. \nPlease set the environment variable and try again: $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"\n or set it in a .key file in the current directory.')
    raise SystemExit(1)

client = genai.Client(api_key=api_key)


# Statikus kontextus és specifikáció beolvasása (csak egyszer fut le)
with open("context.md", "r", encoding="utf-8") as f:
    context_text = f.read()

with open("specification.md", "r", encoding="utf-8") as f:
    spec_text = f.read()

# A System Prompt összeállítása a modell számára
system_prompt = f"{context_text}\n\n{spec_text}"

# 2. Az Ágens-hurok (The Agent Loop)
messages = [
    types.Content(
        role="user",
        parts=[
            types.Part(text="Project context: " + context_text),
            types.Part(text=spec_text),
        ],
    )
]

def agent_loop(system_prompt, tool_config, current_question, rules, answer_text):
    """
    This is the main loop of the agent. 
    It continuously interacts with the model, processes its responses, and executes tools as needed.
    """
    attempts = 0
    while True:
        # Meghívjuk a modellt, és átadjuk neki a futtatható függvényeket
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=messages,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are an automated development agent. "
                            "Specifications: " + system_prompt + "\n"
                        ),
                        tools=[tool_config]
                    )
            )
            attempts = 0
        except Exception as e:
            logger.exception("[ERROR]\tHiba történt - %s/20", attempts)
            attempts += 1
            if attempts >= 20:
                logger.error("[ERROR]\tTúl sok hiba történt, kilépés.")
                break
            continue

        # Ha a modell válaszolni akar a felhasználónak (vége a feladatnak)
        if response.text:
            logger.info("[INFO]\tÁgens válasza: %s", response.text)
            break
            
        # Ha a modell egy eszközt (Function Call) akar meghívni
        if response.function_calls:
            # A modell üzenetét csak egyszer adjuk a history-hoz, különben duplikálódik
            # ha egy válaszban több function call is érkezik.
            messages.append(response.candidates[0].content)
            for call in response.function_calls:
                # 3. Kód szinten elkapjuk a modell szándékát
                tool_name = call.name
                args = call.args
                
                logger.info("[*] Az ágens meg akarja hívni a következőt: %s paraméterekkel: %s", tool_name, args)

                # Dinamikusan lefuttatjuk a valós Python függvényt
                if tool_name == "range_and_type_checker":
                    result = range_and_type_checker(**args)
                elif tool_name == "confidentality_enhance":
                    result = confidentality_enhance(**args)
                else:
                    result = f"Ismeretlen eszköz: {tool_name}"

                logger.info("[INFO]\t[%s] eredmény: %s", tool_name, result)
                    
                # A futási eredményt (visszajelzést) visszacsatoljuk a modellnek
                messages.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=tool_name,
                                response={"result": result},
                            )
                        ],
                    )
                )

############### Orchestration ########################
if __name__ == "__main__":
    tmp = load_the_CSV()
    if tmp is None:
        logger.error("[ERROR]\tNo CSV file to process. Exiting.")
        raise SystemExit(1)
    df = pd.read_csv(tmp)

    scoring_rules = load_the_JSON()
    if scoring_rules is None:
        logger.error("[ERROR]\tNo JSON rule file to process. Exiting.")
        raise SystemExit(1)
    
    results_list = []

    for index, row in df.iterrows():
        print(f"[{index+1}/{len(df)}] Row processing... ")
        
        # Kérdés és szabály kinyerése
        current_question = "question #1"
        rules = scoring_rules["questions"][0][current_question]
        answer_text = row[current_question]
        
        # Ágens meghívása
        final_evaluation = agent_loop(
            system_prompt=system_prompt,
            tool_config=tool_config,
            current_question=current_question,
            rules=rules,
            answer_text=answer_text
        )
        
        # Eredmény elmentése a memóriába (később CSV-be íráshoz)
        results_list.append({
            "original_answer": answer_text,
            "evaluation": final_evaluation # Ide ideálisan egy JSON string érkezik
        })

    # Mentés új CSV-be
    result_df = pd.DataFrame(results_list)
    result_df.to_csv("scored_answers.csv", index=False)