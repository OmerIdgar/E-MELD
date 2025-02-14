from tabulate import tabulate
from utils import load_dataset
from analysis import analysis_main
from approach1 import Approach1, process_data as process_data_approach1
from approach2 import Approach2, process_data as process_data_approach2
from approach3 import Approach3, process_data as process_data_approach3

def main(run_approach1, run_approach2, run_approach3, analyse_results):
    print("Loading dataset...")
    train_df, test_df = load_dataset()
    print("\n" + "=" * 40)
    print("Test Dataset Sample (First 5 Rows)")
    print("=" * 40)
    print(tabulate(test_df.head(), headers="keys", tablefmt="grid"))

    mode = "test"
    max_retries = 20
    num_of_utterances = 5

    # --- Run Approach 1 ---
    if run_approach1:
        approach1_instance = Approach1(max_retries)
        print("Processing data using Approach 1 ...")
        process_data_approach1(mode, approach1_instance, output_file_suffix="approach1", num_of_utterances=num_of_utterances)
        print("Approach 1 processing complete.\n")

    # --- Run Approach 2 ---
    if run_approach2:
        approach2_instance = Approach2(max_retries, is_hash_speakers=False)
        print("Processing data using Approach 2...")
        process_data_approach2(mode, approach2_instance, output_file_suffix="approach2", num_of_utterances=num_of_utterances)

        approach2_instance = Approach2(max_retries, is_hash_speakers=True)
        print("Processing data using Approach 2 - hashed speaker names...")
        process_data_approach2(mode, approach2_instance, output_file_suffix="approach2", num_of_utterances=num_of_utterances)

        print("Approach 2 processing complete.\n")

    # --- Run Approach 3 ---
    if run_approach3:
        approach3_instance = Approach3(max_retries, is_hash_speakers=False)
        print("Processing data using Approach 3...")
        process_data_approach3(mode, approach3_instance, output_file_suffix="approach3", group_by=["Episode", "Season"],num_of_utterances=num_of_utterances)

        approach3_instance = Approach3(max_retries, is_hash_speakers=True)
        print("Processing data using Approach 3 - hashed speaker names...")
        process_data_approach3(mode, approach3_instance, output_file_suffix="approach3", group_by=["Episode", "Season"], num_of_utterances=num_of_utterances)

        print("Approach 3 processing complete.")

    if not analyse_results:
        return
    
    analysis_main()


if __name__ == "__main__":
    run_approach1 = True
    run_approach2 = True
    run_approach3 = True
    analyse_results = True
    main(run_approach1, run_approach2, run_approach3, analyse_results)
