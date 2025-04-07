import json
import openai


OPENAI_API_KEY = "api_key_here"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_obd2_diagnostics(code, description):
    print(f"Fetching OBD-II data for {code}...")

    prompt = f"""
    Provide structured diagnostic information for the OBD-II trouble code {code}: {description}.
    Return only valid JSON, with no extra text or formatting.
    Use this exact format:
    {{
        "code": "{code}",
        "description": "{description}",
        "possible_causes": ["Cause 1", "Cause 2", "Cause 3"],
        "diagnostic_steps": ["Step 1", "Step 2", "Step 3"]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content.strip()

    # Remove Markdown formatting if present
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()

    try:
        parsed_data = json.loads(content)
    except json.JSONDecodeError:
        print(f"Error parsing JSON for code {code}: {content}")
        return {"code": code, "error": "Invalid JSON response from OpenAI"}

    return parsed_data

# Read OBD-II codes from file
with open("obd2_codes.txt", "r") as file:
    codes = [line.strip().split(" - ", 1) for line in file]

# Generate diagnostics for all codes
print("Data TRY")
obd2_data = {code[0]: get_obd2_diagnostics(code[0], code[1]) for code in codes}
print("Data Gotten")

# Save to a structured JSON file
with open("obd2_data.json", "w") as json_file:
    json.dump(obd2_data, json_file, indent=4)

print("OBD-II diagnostic data collected successfully!")

