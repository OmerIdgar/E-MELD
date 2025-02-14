import os
import re
import tqdm
import pandas as pd
from ollama_setup import run_llm
from config import TRAIN_PATH, TEST_PATH, FINAL_SAVE_DIR, TEMP_FINAL_SAVE_DIR

class Approach1:
    def __init__(self, max_retries=5):
        self.max_retries = max_retries

        # Updated roles description for a single utterance.
        self.roles_description = """
You are an expert in analyzing a single utterance and assigning the appropriate speaker role.

The roles are:
- Protagonist: Leads the discussion and asserts authority, actively driving the conversation forward.
- Supporter: Offers encouragement, help, or positive reinforcement to the speaker.
- Neutral: Participates passively, often giving straightforward responses or reacting without adding substantial direction or conflict.
- Gatekeeper: Facilitates smooth communication and guides turn-taking, ensuring clarity.
- Attacker: Challenges ideas or expresses skepticism, introducing tension or conflict.

For the given utterance, output only the speaker's role and a brief justification.
Do not include any additional commentary or formatting outside of the specified JSON format.

Format the output as a JSON object with the following keys:
"Speaker": "<speaker name>",
"Role": "<chosen role>",
"Justification": "<detailed justification referencing the utterance and context>"
        """

    def generate_prompt(self, conversation, sr_no_list, dialogue_id):
        """
        Build a prompt for a single utterance.
        Assumes that conversation is a list containing one tuple (speaker, utterance).
        """
        speaker, utterance = conversation[0]
        conversation_text = f"Sr No. {sr_no_list[0]}, {speaker}: \"{utterance}\""
        prompt = (
            f"{self.roles_description}\n\n"
            f"Here is the utterance from Dialogue_ID {dialogue_id}:\n"
            f"{conversation_text}\n\n"
            "Identify the role for the speaker above and provide a brief justification. "
            "Include the length (in characters) of the utterance in your justification."
        )
        return prompt

    def assign_roles(self, conversation, sr_no_list, speakers_list, dialogue_id):
        """
        For the given single utterance (and its associated speaker and serial number),
        builds the prompt, calls the LLM with retry logic, and returns the parsed result.
        """
        prompt = self.generate_prompt(conversation, sr_no_list, dialogue_id)
        template = (
            "You are an assistant specialized in analyzing a single utterance and identifying the speaker's role. "
            "Answer the following question in a valid JSON format.\n\n"
            "Question: {question}\n\n"
            "Answer: Think step by step and provide a JSON object following the specified format."
        )
        attempts = 0
        response = None
        while attempts < self.max_retries:
            try:
                response = run_llm(prompt, template)
                parsed_results = self.parse_response(response, sr_no_list, speakers_list, prompt, dialogue_id)
                if parsed_results and parsed_results[0]["Role"] != "Error":
                    # Annotate additional fields.
                    for result in parsed_results:
                        result["Dialogue_ID"] = dialogue_id
                        result["Prompt"] = prompt
                        result["Response"] = response
                    return parsed_results
            except Exception as e:
                # Do not catch critical errors like connection refusal.
                if "Connection refused" in str(e):
                    raise RuntimeError(f"Critical Error: {str(e)}") from e
            attempts += 1

        print(f"Failed to assign role for Dialogue_ID {dialogue_id}, Sr No. {sr_no_list[0]} after {self.max_retries} attempts.", flush=True)
        return [{
            "Sr No.": sr_no_list[0],
            "Speaker": speakers_list[0],
            "Dialogue_ID": dialogue_id,
            "Role": "Error",
            "Justification": f"Failed after {self.max_retries} attempts.",
            "Prompt": prompt,
            "Response": response
        }]

    def parse_response(self, response, sr_no_list, speakers_list, prompt, dialogue_id):
        """
        Parses the LLM's JSON response for a single utterance.
        Expects keys "Speaker", "Role", and "Justification".
        """
        speaker_match = re.search(r'"Speaker":\s*"([^"]+)"', response)
        role_match = re.search(r'"Role":\s*"([^"]+)"', response)
        justification_match = re.search(r'"Justification":\s*"([^"]+)"', response)

        speaker = speaker_match.group(1) if speaker_match else (speakers_list[0] if speakers_list else "Unknown")
        role = role_match.group(1) if role_match else "Error"
        justification = justification_match.group(1) if justification_match else "Error parsing response."

        result = {
            "Sr No.": sr_no_list[0],
            "Speaker": speaker,
            "Dialogue_ID": dialogue_id,
            "Role": role,
            "Justification": justification,
            "Prompt": prompt,
            "Response": response
        }
        return [result]


def process_data(mode, model_instance: Approach1, output_file_suffix, num_of_utterances=None):
    """
    Processes input CSV data for Approach1 in baseline mode.
    Instead of grouping by dialogue, each utterance is processed individually.
    The result (including Sr No., Speaker, Dialogue_ID, Role, and Justification) is saved to a CSV file.
    """
    input_path = TRAIN_PATH if mode == 'train' else TEST_PATH
    input_df = pd.read_csv(input_path)
    input_df = input_df[['Sr No.', 'Dialogue_ID', 'Speaker', 'Utterance']]

    if num_of_utterances:
        input_df = input_df.head(num_of_utterances)

    output_file = os.path.join(
        TEMP_FINAL_SAVE_DIR,
        f"{mode}_{output_file_suffix}.csv"
    )

    # Create the output file with header if it does not exist.
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Sr No.,Speaker,Dialogue_ID,Role,Justification,Prompt,Response\n")

    try:
        existing_df = pd.read_csv(output_file)
    except pd.errors.EmptyDataError:
        existing_df = pd.DataFrame(columns=["Sr No.", "Speaker", "Dialogue_ID", "Role", "Justification", "Prompt", "Response"])

    # Process each utterance individually.
    for index, row in tqdm.tqdm(input_df.iterrows(), total=len(input_df), desc="Processing utterances"):
        sr_no = row['Sr No.']
        dialogue_id = row['Dialogue_ID']
        speaker = row['Speaker']
        utterance = row['Utterance']

        # Skip if this utterance was already processed.
        if not existing_df.empty:
            if ((existing_df["Dialogue_ID"] == dialogue_id) & (existing_df["Sr No."] == sr_no)).any():
                continue

        conversation_data = [(speaker, utterance)]
        sr_no_list = [sr_no]
        speakers_list = [speaker]

        roles = model_instance.assign_roles(conversation_data, sr_no_list, speakers_list, dialogue_id)
        res_df = pd.DataFrame(roles)
        existing_df = pd.concat([existing_df, res_df]).sort_values(by="Sr No.").reset_index(drop=True)
        existing_df.to_csv(output_file, index=False, encoding='utf-8')
    existing_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Results saved to {output_file}")
    return existing_df
