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
# 1. Define the tools that the agent can use
def range_and_type_checker(score: float, range_min: float = 0.0, range_max: float = 10.0) -> str:
    """
    If the agent provides a value outside the scale, this tool immediately forces the model to 
    regenerate the output with an error message 'The score is not within the specified range; please try again.' 
    before the data is written to the output file.
    """
    try:
        if not isinstance(score, (int, float)):
            return "[ERROR] The score is not a number; please try again."
        if not (range_min <= score <= range_max):
            return "[ERROR] The score is not within the specified range(%s, %s); please try again." % (range_min, range_max)
        return "[INFO] The score is valid."
    except Exception as e:
        return f"[ERROR] An unexpected error occurred: {e}"

def confidentality_enhance(current_score: float, confidence: float, reasoning: str) -> str:
    """
    If the model is uncertain (confidence below 0.5), 
    we do not accept the low-quality prediction; 
    instead, we initiate a guided re-evaluation (Self-Refine) loop.
    """
    if confidence < 0.5:
        return f"[WARNING] The model's confidence is low ({confidence}); initiating a guided Self-Refine loop. Reasoning: {reasoning}. Please re-read the respondent's texts and provide a new evaluation. Focus in whether the response contains the key points and whether it is relevant to the question. Please provide a new score and reasoning. If the issue persists, use a lower score, but with at least 0.51 confidence value."
    else:
        return f"[INFO] The model's confidence is sufficient ({confidence}); no Self-Refine loop needed. Reasoning: {reasoning}."
    
# Secure tool registration
my_tools = [range_and_type_checker, confidentality_enhance]
tool_config = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="range_and_type_checker",
            description="If the agent provides a value outside the scale, this tool immediately forces the model to regenerate the output.",
            parameters={
                "type": "object",
                "properties": {
                    "score": {"type": "number", 
                              "description": "This function checks if the score is within the specified range and of the correct type."}
                },
                "required": ["score"],
            },
        ),
        types.FunctionDeclaration(
            name="confidentality_enhance",
            description="It's necessary to use it if the model is uncertain, the confidence is lower then 0.5, to initiate a guided Self-Refine loop.",
            parameters={
                "type": "object",
                "properties": {
                    "current_score": {"type": "number",
                                      "description": "The curreent score provided by the model, which is below the confidence threshold of 0.5."},
                    "confidence": {"type": "number",
                                   "description": "The confidence level of your evaluation, which is below the threshold of 0.5, indicating uncertainty in the model's prediction."},
                    "reasoning": {"type": "string",
                                   "description": "The reasoning behind your evaluation, explaining why the model's confidence is low and why a Self-Refine loop is necessary."}              
                },
                "required": ["current_score", "confidence", "reasoning"],
            },
        ),
    ]
)

# The final output schema that the model must return
final_output_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "score": types.Schema(type=types.Type.NUMBER),
        "confidence": types.Schema(type=types.Type.NUMBER),
        "reasoning": types.Schema(type=types.Type.STRING),
    },
    required=["score", "confidence", "reasoning"]
)

# Load static context and specification (only runs once)
with open("context.md", "r", encoding="utf-8") as f:
    context_text = f.read()

with open("specification.md", "r", encoding="utf-8") as f:
    spec_text = f.read()

# Compose the System Prompt for the model
system_prompt = f"{context_text}\n\n{spec_text}"

# 2. The Agent Loop
def agent_loop(llm_model, system_prompt, tool_config, current_question, rules, range_min, range_max, answer_text, maximum_attenpts=20):
    """
    This is the main loop of the agent. 
    It continuously interacts with the model, processes its responses, and executes tools as needed.
    """

    user_prompt = f"Question: {current_question}\nRules: {rules}\nAnswer to evaluate: {answer_text}"
    
    messages = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_prompt)
            ],
        )
    ]

    attempts = 0

    while True:
        # Call the model and pass it the callable functions
        try:
            response = client.models.generate_content(
                model=llm_model,
                contents=messages,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=[tool_config],
                        response_mime_type="application/json",
                        response_schema=final_output_schema,
                        temperature=0.1
                    )
            )
            attempts = 0
        except Exception as e:
            logger.exception("[ERROR]\tException during model call \t%s/%s", attempts, maximum_attenpts)
            attempts += 1
            if attempts >= maximum_attenpts:
                logger.error("[ERROR]\tToo many failed attempts, exiting from agent loop.")
                return None
            continue

        # If the model wants to respond to the user (task is complete)
        if response.text:
            logger.info("[INFO]\tAgent's answer: %s", response.text)
            return response.text # Must return the result to the main program
            
        # If the model wants to call a tool (Function Call)
        if response.function_calls:
            # Add the model's message to history only once to avoid duplication
            # if multiple function calls arrive in one response
            messages.append(response.candidates[0].content)
            for call in response.function_calls:
                # 3. Capture the model's intention at the code level
                tool_name = call.name
                args = call.args
                
                logger.info("[*]\tAgent tool-call: %s\t with args: %s", tool_name, args)

                # Dynamically execute the actual Python function
                if tool_name == "range_and_type_checker":
                    result = range_and_type_checker(**args, range_min=range_min, range_max=range_max)
                elif tool_name == "confidentality_enhance":
                    result = confidentality_enhance(**args)
                else:
                    result = f"Unknown tool: {tool_name}"

                logger.info("[INFO]\t[%s] result:\t%s", tool_name, result)
                    
                # Feed back the execution result (feedback) to the model
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
    logger.info("[SYSTEM]\tStarting the Interview Scoring System...")

    # Startup working directory
    startup_cwd = Path.cwd().resolve()
    logger.info("[INFO]\tStartup working directory: %s", startup_cwd)

    # API key
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = try_load_api_key_from_key_file()

    if not api_key:
        logger.error('[FATAL ERROR]\tGEMINI_API_KEY is not set. Stopping program. \nPlease set the environment variable and try again: $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"\n or set it in a .key file in the current directory.')
        raise SystemExit(1)

    client = genai.Client(api_key=api_key)

    # Load input data
    df = load_the_CSV()
    if df is None or df.empty:
        logger.error("[FATAL ERROR]\tNo CSV file to process. Exiting.")
        raise SystemExit(1)
    df.columns = [str(col).strip(' "') for col in df.columns]

    scoring_rules = load_the_JSON()
    if scoring_rules is None:
        logger.error("[FATAL ERROR]\tNo JSON rule file to process. Exiting.")
        raise SystemExit(1)
    rules_dict = {}
    for q_obj in scoring_rules["questions"]:
        for question_key, rule_data in q_obj.items():
            rules_dict[question_key] = rule_data

    results_list = []
    maximum_attenpts = 20  # set this higher for more attempts in case of model errors, but it will take longer to process the CSV file
    llm_models = ['gemini-3.5-flash', 'gemini-2.5-flash', 'gemini-1.5-flash']  # choose your preferred llm model
    output_file = "scored_answers.csv"

    for position, (index, row) in enumerate(df.iterrows(), start=1):
        print(f"[{position}/{len(df)}]\tRow processing...")

        for question_column in df.columns:
            if question_column in rules_dict:
                rule_list = rules_dict[question_column]
                scoring_method = rule_list[0]
                range_min = rule_list[1]
                range_max = rule_list[2]

                answer_text = row.get(question_column, "")
                if pd.isna(answer_text) or str(answer_text).strip() == "":
                    logger.warning("[WARNING]\tRow %d, Question '%s' has no answer. Skipping.", position + 1, question_column)
                    continue

                safe_answer_text = security.main_security_sanitization(str(answer_text))
                
                # Call the agent
                final_evaluation = agent_loop(
                    llm_model=llm_models[0],  # You can change the model here if needed
                    system_prompt=system_prompt,
                    tool_config=tool_config,
                    current_question=question_column,
                    rules=scoring_method,
                    range_min=range_min,
                    range_max=range_max,
                    answer_text=safe_answer_text,
                    maximum_attenpts=maximum_attenpts
                )

                if final_evaluation is None:
                    logger.error("[ERROR]\tRow %d, Question '%s': Agent failed to provide a valid evaluation after %d attempts. Skipping.", position + 1, question_column, maximum_attenpts)
                    continue
                else:
                    # 1. Convert the JSON string to a Python dictionary
                    try:
                        eval_data = json.loads(final_evaluation)
                    except json.JSONDecodeError as e:
                        logger.error("[ERROR]\tRow %d, Question '%s': Failed to parse JSON response. Raw output: %s", position + 1, question_column, final_evaluation)
                        results_list.append({
                                                "respondent_row": position + 1,
                                                "question": question_column,
                                                "original_answer": safe_answer_text,
                                                "raw_evaluation": final_evaluation
                                            })
                        continue

                    # 2. Save the extracted parameters to separate keys (columns)
                    results_list.append({
                        "respondent_row": position + 1,
                        "question": question_column,
                        "original_answer": safe_answer_text,
                        "score": eval_data.get("score"),
                        "confidence": eval_data.get("confidence"),
                        "reasoning": eval_data.get("reasoning")
                    })
            else:
                logger.warning("[WARNING]\tRow %d, Question '%s' is not defined in the JSON rules. Skipping.", position + 1, question_column)
                continue


    # Save to new CSV    
    result_df = pd.DataFrame(results_list)
    result_df.to_csv(output_file, index=False)
    logger.info("[SYSTEM]\tResults saved to '%s'", output_file)
    logger.info("[SYSTEM]\tExecution completed successfully.")