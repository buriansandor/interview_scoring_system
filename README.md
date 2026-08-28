# Interview answer scoring system
This is a tool to give score values to text-based answers from interviews. This is useful in the evaluation of surveys.

The goal of this scirpt is to make it possible to evaluate several answers with statistical methods, converting the text to numeric values in specidic ranges.



## How-to

Requirements:

- [Python 3.11+](https://www.python.org/)
- A [Google Gemini API key](https://aistudio.google.com/api-keys)

### 1. Activate Virtual Environment

PowerShell:
```
py -m venv .venv; . .\.venv\Scripts\Activate.ps1
```

or

```powershell
.\.venv\Scripts\Activate.ps1
```

If needed (temporary for current shell only):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Install Dependencies
```shell
pip install -r requirements.txt
```

or manually

```powershell
pip install google-genai
pip install pandas
```

### 3. Set Environment Variable

Set your API key in a file which has extension `.key`

### 4. Run the Agent

```powershell
python .\app.py
```

## Input
1. Survey answer dataset as `csv`
2. The scoring method as a `json`

The survey answers are an input from `csv` file, where the first row's columns are the questions, and every other row is an answer from a person.

The scoring method is a JSON file, where the `questions` list has objects as elements, each object's key parameter is the question's string, the same one as from the `csv` file.


Example structure of the JSON file:
````
{
    "questions": 
    [
        {"question #1": [
            "scoring method, to-dos",
            range_start_value_as_number, range_end_value_as_number]
        }
    ]
}
````

The input can be set next to the script in the same folder, or if the script won't see it, then it will ask for it in the terminal.

Example input can be the [evaluation.json](evaluation.json) and the [test.csv](test.csv).

## The output

The output by default is `scored_answers.csv` and `agent.log`. The structure of the output:
````
respondent_row,question,original_answer,score,confidence,reasoning
````

__________________________________________________________________

## About the script

````Mermaid

flowchart TD
    Start([Start the script]) --> Load[Answers CSV and scoring method JSON]
    
    Load --> RowLoop{Is there unprocessed answer in the CSV file}
    RowLoop -- No --> SaveFinal[Save the output data in CSV] --> Finish([Exit])
    
    RowLoop -- Yes --> QLoop{Have we finished with the given row?}
    QLoop -- Yes --> NextRow[Next row] --> RowLoop
    
    QLoop -- No --> ExtractData[Get the current scoring rules for the given question]
    ExtractData --> AgentCall[Call the LLM Agent]
    
    AgentCall --> OutputObject[Output of the agent:\n- score\n- confidence\n- reasoning]
    
    OutputObject --> RangeCheck{range_and_type_checker tool:\nvalidation of the answer}
    
    RangeCheck -- Not valid --> RangeError[Send back with error message] --> AgentCall
    
    RangeCheck -- Valid --> ConfidenceCheck{confidence < 0.5?}
    
    ConfidenceCheck -- Yes --> EnhanceTool[confidentality_enhance:\nBased on the 'reasoning' values of the answer,\n new question about the same entry, with the reason and score] --> AgentCall
    
    ConfidenceCheck -- No --> SaveScore[Save the score value]
    SaveScore --> QLoop

    %% Stílusok a jobb átláthatóságért
    classDef loop fill:#f9f2f4,stroke:#333,stroke-width:2px;
    classDef agent fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef tool fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    
    class RowLoop,QLoop loop;
    class AgentCall,OutputObject agent;
    class RangeCheck,ConfidenceCheck,EnhanceTool,RangeError tool;

````

The LLM is a Gemini-based by default.

The answer of the LLM is an object, it hase these parameters:
- score - value between the given range
- confidence - value between 0.0-1.0
- reasoning - the reason of the decision

Available tools for the AI Agent:
- range_and_type_checker - validaton
- confidentality_enhance - if the confidence value is lower then 0.5 then this tool will give a second wuestion based on the reason of the score and the scoring method, to give a value  with better confidentality. This is the self-reflection.

The agent get the data one-by one, row by row.