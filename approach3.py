import os
import tqdm
import pandas as pd
from ollama_setup import run_llm
from base_role_approach import BaseRoleApproach
from config import TRAIN_PATH, TEST_PATH, FINAL_SAVE_DIR, TEMP_FINAL_SAVE_DIR

class Approach3(BaseRoleApproach):
    """
    Extends BaseRoleApproach to build a prompt that includes each utterance’s duration
    and an optional connection summary of how speakers interact.
    """
    def __init__(self, max_retries=5, is_hash_speakers=False):
        super().__init__(max_retries=max_retries, is_hash_speakers=is_hash_speakers)
        self.roles_description = (
            "\nYou are an expert in analyzing conversations and assigning speaker roles.\n"
            "The following conversation is taken from various contexts, and your task is to assign roles "
            "to each speaker based on their utterances, including their role in the overall dialogue.\n\n"
            "The only roles possible are:\n"
            "- Protagonist: Leads the discussion and asserts authority, actively driving the conversation forward.\n"
            "- Supporter: Offers encouragement, help, or positive reinforcement to other speakers.\n"
            "- Neutral: Participates passively, often giving straightforward responses or reacting without adding substantial direction or conflict.\n"
            "- Gatekeeper: Facilitates smooth communication, guides turn-taking, or helps clarify misunderstandings, ensuring the conversation remains balanced and productive.\n"
            "- Attacker: Challenges others, expresses skepticism, or undermines the ideas and confidence of other speakers, introducing tension or conflict into the interaction.\n\n"
            "Format the output as a JSON file with the following structure, for each utterance in the dialogue:\n"
            "{\n"
            '\t"Sr No.": <Sr No.>,\n'
            '\t"Speaker": <Speaker_Name>,\n'
            '\t"Role": <Chosen_Role>,\n'
            '\t"Justification": <Detailed_Reason>\n'
            "}"
        )

    def generate_prompt(self, conversation, sr_no_list, duration_list, dialogue_id, speakers_list, connection_summary=None):
        """
        Generates the dialogue prompt.
        Includes utterance durations and (optionally) connection summaries.
        """
        # If hashing is enabled, replace speaker names with hashed identifiers.
        if self.is_hash_speakers:
            hashed_speakers_list, _ = self._hash_speakers(speakers_list)
        else:
            hashed_speakers_list = speakers_list

        conversation_text = "\n".join(
            f"Sr No. {sr_no}, {hashed_speaker} ({round(duration, 2)}s): \"{utterance}\""
            for sr_no, duration, (speaker, utterance), hashed_speaker
            in zip(sr_no_list, duration_list, conversation, hashed_speakers_list)
        )

        connection_text = ""
        if connection_summary:
            connection_text = "\n\nHere is a detailed summary of how the participants interact with specific individuals:\n"
            connection_text += "\n".join(
                f"- Speaker `{conn['Speaker_Response']}` when interacting with `{conn['Speaker_Responded_To']}`:\n"
                f"  • Avg Duration of Response: {conn['Response_Duration']:.2f}s\n"
                f"  • Avg Words Used: {conn['Words_in_Response']:.2f}\n"
                f"  • Avg Letters Used: {conn['Letters_in_Response']:.2f}\n"
                f"  • Sentiments Expressed: {', '.join([f'{sent[0]} ({sent[1]}%)' for sent in conn['Sentiment_in_Response']])}\n"
                f"  • Emotions Displayed: {', '.join([f'{emo[0]} ({emo[1]}%)' for emo in conn['Emotion_in_Response']])}"
                for conn in connection_summary.get('Connection_Summary', [])
            )
            connection_text += "\n\nNow, let's look at an overall view of how each speaker communicates in general, across all their conversations:\n"
            connection_text += "\n".join(
                f"- Speaker `{part['Speaker_Response']}` (overall communication):\n"
                f"  • Avg Duration of Response: {part['Response_Duration']:.2f}s\n"
                f"  • Avg Words Used: {part['Words_in_Response']:.2f}\n"
                f"  • Avg Letters Used: {part['Letters_in_Response']:.2f}\n"
                f"  • Sentiments Expressed: {', '.join([f'{sent[0]} ({sent[1]}%)' for sent in part['Sentiment_in_Response']])}\n"
                f"  • Emotions Displayed: {', '.join([f'{emo[0]} ({emo[1]}%)' for emo in part['Emotion_in_Response']])}"
                for part in connection_summary.get('Participants_Summary', [])
            )
            connection_text += (
                "\n\nKey Insight:\n"
                "The `Connection Summary` provides a focused view of how speakers communicate with specific individuals. "
                "In contrast, the `Participants Summary` reveals their general communication patterns across all interactions.\n"
                "Comparing these summaries can highlight whether speakers adjust their communication style based on the person they are speaking to."
            )

        prompt = (
            f"{self.roles_description}\n\n"
            f"Here is the context of the entire dialogue with Dialogue_ID {dialogue_id}:\n{conversation_text}\n"
            f"{connection_text}\n\n"
            "Identify the role and provide justifications for each speaker. Ensure each response includes "
            "'Sr No.', 'Speaker', 'Role', and 'Justification'. Additionally, include the length of each dialogue utterance in the response."
        )
        return prompt

    def assign_roles(self, conversation, sr_no_list, duration_list, speakers_list, dialogue_id, connection_summary=None):
        """
        Builds the prompt (with durations and connection summaries), calls the LLM with retry logic,
        and returns parsed results.
        """
        prompt = self.generate_prompt(conversation, sr_no_list, duration_list, dialogue_id, speakers_list, connection_summary)
        if self.is_hash_speakers:
            speakers_for_validation, _ = self._hash_speakers(speakers_list)
        else:
            speakers_for_validation = speakers_list

        template = (
            "You are an assistant specialized in analyzing dialogue and identifying speaker roles. "
            "Answer the following question in a valid JSON format.\n\n"
            "Question: {question}\n\n"
            "Answer: Think step by step and provide a JSON object following the specified format."
        )

        attempts = 0
        response = None
        while attempts < self.max_retries:
            try:
                response = run_llm(prompt, template)
                parsed_results = self.parse_response(response, sr_no_list, speakers_for_validation, prompt, dialogue_id)
                if all(result["Role"] != "Error" for result in parsed_results):
                    for result in parsed_results:
                        result["Dialogue_ID"] = dialogue_id
                        result["Prompt"] = prompt
                        result["Response"] = response
                    return parsed_results
            except Exception as e:
                if "Connection refused" in str(e):
                    raise RuntimeError(f"Critical Error: {str(e)}") from e
            attempts += 1

        print(f"Failed to assign roles for Dialogue_ID {dialogue_id} after {self.max_retries} attempts.", flush=True)
        return [{
            "Sr No.": sr_no,
            "Speaker": speaker,
            "Dialogue_ID": dialogue_id,
            "Role": "Error",
            "Justification": f"Failed after {self.max_retries} attempts.",
            "Prompt": prompt,
            "Response": response
        } for sr_no, speaker in zip(sr_no_list, speakers_for_validation)]

def process_data(mode, model_instance: Approach3, output_file_suffix, group_by, num_of_utterances=None):
    """
    Loads the CSV data (with time fields), computes durations, groups by Dialogue_ID,
    obtains a connection summary for each dialogue, and calls the model to assign roles.
    """
    input_path = TRAIN_PATH if mode == 'train' else TEST_PATH

    # Read the raw CSV and make a copy for processing.
    original_df = pd.read_csv(input_path)
    input_df = original_df.copy()
    input_df['StartTime'] = pd.to_datetime(input_df['StartTime'].str.replace(',', '.'), format="%H:%M:%S.%f")
    input_df['EndTime'] = pd.to_datetime(input_df['EndTime'].str.replace(',', '.'), format="%H:%M:%S.%f")
    input_df['Duration'] = (input_df['EndTime'] - input_df['StartTime']).dt.total_seconds()
    # Ensure these fields are available for summarization.
    input_df = input_df[['Sr No.', 'Dialogue_ID', 'Speaker', 'Utterance', 'Duration',
                         'Season', 'Episode', 'Utterance_ID', 'StartTime', 'EndTime']]

    if num_of_utterances:
        input_df = input_df.head(num_of_utterances)

    grouped_conversations = input_df.groupby('Dialogue_ID')

    output_file = os.path.join(
        TEMP_FINAL_SAVE_DIR,
        f"{mode}_{output_file_suffix}{'_hashed' if model_instance.is_hash_speakers else ''}.csv"
    )
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Sr No.,Speaker,Dialogue_ID,Role,Justification,Prompt,Response\n")
    try:
        existing_df = pd.read_csv(output_file)
    except pd.errors.EmptyDataError:
        existing_df = pd.DataFrame(
            columns=["Sr No.", "Speaker", "Dialogue_ID", "Role", "Justification", "Prompt", "Response"]
        )

    # Compute the response DataFrame used for connection summaries.
    response_df = get_response_df(pd.read_csv(TRAIN_PATH))

    for dialogue_id, group in tqdm.tqdm(grouped_conversations, desc="Processing Dialogues"):
        # Skip dialogue if already processed successfully.
        if dialogue_id in set(existing_df.get("Dialogue_ID", [])):
            dialogue_entries = existing_df[existing_df["Dialogue_ID"] == dialogue_id]
            if "Error" not in dialogue_entries["Role"].values:
                continue
            existing_df = existing_df[existing_df["Dialogue_ID"] != dialogue_id]

        conversation_data = list(zip(group['Speaker'], group['Utterance']))
        sr_no_list = group['Sr No.'].tolist()
        duration_list = group['Duration'].tolist()
        speakers_list = group['Speaker'].tolist()

        try:
            dialogue_connection_summary = summarize_dialogue_connections(
                response_df,
                original_df,
                dialogue_id,
                group_by=group_by,
                is_hash_speakers=model_instance.is_hash_speakers
            )
        except Exception as e:
            dialogue_connection_summary = None
            print(f"Error summarizing connections for Dialogue_ID {dialogue_id}: {e}")

        roles = model_instance.assign_roles(
            conversation_data,
            sr_no_list,
            duration_list,
            speakers_list,
            dialogue_id,
            connection_summary=dialogue_connection_summary
        )
        res_df = pd.DataFrame(roles)
        existing_df = pd.concat([existing_df, res_df]).sort_values(by="Sr No.").reset_index(drop=True)
        existing_df.to_csv(output_file, index=False, encoding='utf-8')
    existing_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Results saved to {output_file}")




###############################################################################
# Helper functions for connection summaries
###############################################################################

def get_response_df(train_df):
    train_df = train_df.copy()
    df = train_df.sort_values(by=['Dialogue_ID', 'Utterance_ID']).reset_index(drop=True)

    def convert_if_time_only(val):
        val = str(val).strip()  # Ensure it's a string and remove leading/trailing spaces
        if ":" in val and "-" not in val:  # Time-only format (e.g., 0:20:40,614)
            return pd.to_datetime(val.replace(',', '.'), format='%H:%M:%S.%f')
        else:  # Already in full datetime format
            return pd.to_datetime(val, errors='coerce')  # Convert safely

    df['StartTime'] = df['StartTime'].apply(convert_if_time_only)
    df['EndTime'] = df['EndTime'].apply(convert_if_time_only)

    response_data = []

    for dialogue_id, group in df.groupby('Dialogue_ID'):
        group = group.reset_index(drop=True)

        for i in range(len(group) - 1):
            pair_0 = group.iloc[i]
            pair_1 = group.iloc[i + 1]

            if pair_1['Dialogue_ID'] == pair_0['Dialogue_ID']:
                response_duration = (pair_1['EndTime'] - pair_1['StartTime']).total_seconds()
                response = {
                    'Sr No.': pair_1['Sr No.'],
                    'Speaker_Response': pair_1['Speaker'],
                    'Speaker_Responded_To': pair_0['Speaker'],
                    'Words_in_Response': len(str(pair_1['Utterance']).split()),
                    'Letters_in_Response': len(str(pair_1['Utterance'])),
                    'Sentiment_in_Response': pair_1['Sentiment'],
                    'Emotion_in_Response': pair_1['Emotion'],
                    'Response_Duration': response_duration,
                    'Dialogue_ID': pair_1['Dialogue_ID'],
                    'Utterance_ID': pair_1['Utterance_ID'],
                    'Season': pair_1['Season'],
                    'Episode': pair_1['Episode'],
                    'StartTime': pair_1['StartTime'],
                    'EndTime': pair_1['EndTime']
                }
                response_data.append(response)

    response_df = pd.DataFrame(response_data)
    return response_df


def hash_speakers_list(speakers_list):
    """
    Given a list of speakers in order of appearance, returns a mapping and a hashed list.
    """
    mapping = {}
    hashed_list = []
    next_char = ord('A')
    for speaker in speakers_list:
        if speaker not in mapping:
            mapping[speaker] = f"Person {chr(next_char)}"
            next_char += 1
        hashed_list.append(mapping[speaker])
    return mapping, hashed_list


def summarize_dialogue_connections(response_df, input_data, dialogue_id, group_by, is_hash_speakers=False):
    response_df = response_df.copy()
    input_data = input_data.copy()
    input_data_response_df = get_response_df(input_data)
    dialogue_data = input_data_response_df[input_data_response_df['Dialogue_ID'] == dialogue_id]

    if dialogue_data.empty:
        raise ValueError(f"No data found for Dialogue_ID: {dialogue_id}")

    season, episode = input_data[['Season', 'Episode']].iloc[0]

    grouped_data = response_df[(response_df[group_by] == dialogue_data[group_by].iloc[0]).all(axis=1)] if group_by != 'None' else response_df
    actual_pairs = dialogue_data[['Speaker_Response', 'Speaker_Responded_To']].drop_duplicates()
    actual_speakers = dialogue_data['Speaker_Response'].drop_duplicates()
    connections = grouped_data.merge(actual_pairs, on=['Speaker_Response', 'Speaker_Responded_To'])
    connections = connections[connections['Speaker_Response'] != connections['Speaker_Responded_To']]
    speakers = grouped_data.merge(actual_speakers, on=['Speaker_Response'])
    def top_3_aggregate(series):
        counts = series.value_counts(normalize=True).drop(labels='neutral', errors='ignore') * 100
        return [(val, round(perc, 2)) for val, perc in counts.head(3).items()]

    # Connection Summary
    connection_summary = connections.groupby(['Speaker_Response', 'Speaker_Responded_To']).agg(
        Response_Duration=('Response_Duration', 'mean'),
        Words_in_Response=('Words_in_Response', 'mean'),
        Letters_in_Response=('Letters_in_Response', 'mean'),
        Sentiment_in_Response=('Sentiment_in_Response', lambda x: top_3_aggregate(x)),
        Emotion_in_Response=('Emotion_in_Response', lambda x: top_3_aggregate(x))
    ).reset_index()

    # Participants Summary
    participants_summary = speakers.groupby('Speaker_Response').agg(
        Response_Duration=('Response_Duration', 'mean'),
        Words_in_Response=('Words_in_Response', 'mean'),
        Letters_in_Response=('Letters_in_Response', 'mean'),
        Sentiment_in_Response=('Sentiment_in_Response', lambda x: top_3_aggregate(x)),
        Emotion_in_Response=('Emotion_in_Response', lambda x: top_3_aggregate(x))
    ).reset_index()

    summary = {
        'Dialogue_ID': dialogue_id,
        'Season': season,
        'Episode': episode,
        'Participants': dialogue_data['Speaker_Response'].unique().tolist(),
        'Connection_Summary': connection_summary.to_dict(orient='records'),
        'Participants_Summary': participants_summary.to_dict(orient='records')
    }

    if is_hash_speakers:
        # Create a mapping for all speakers appearing in this dialogue (both as responders and responded-to)
        speakers_all = list(set(
            list(dialogue_data['Speaker_Response'].unique()) +
            list(dialogue_data['Speaker_Responded_To'].unique())
        ))
        mapping, _ = hash_speakers_list(speakers_all)

        # Update Connection Summary: replace speaker names with hashed ones.
        for conn in summary['Connection_Summary']:
            original_resp = conn['Speaker_Response']
            original_responded = conn['Speaker_Responded_To']
            conn['Speaker_Response'] = mapping.get(original_resp, original_resp)
            conn['Speaker_Responded_To'] = mapping.get(original_responded, original_responded)

        # Update Participants Summary: replace speaker names with hashed ones.
        for part in summary['Participants_Summary']:
            original = part['Speaker_Response']
            part['Speaker_Response'] = mapping.get(original, original)

        # Update list of Participants
        summary['Participants'] = [mapping.get(sp, sp) for sp in summary['Participants']]

    return summary
