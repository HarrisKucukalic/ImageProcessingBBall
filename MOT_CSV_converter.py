import pandas as pd
import os
# csv and video from https://www.kaggle.com/datasets/atomscott/teamtrack?resource=download
def convert_csv_to_mot_format(csv_path, output_path):
    """
    Converts a ground truth CSV with a multi-level header (wide format)
    into the standard MOT Challenge format (long format).

    The MOT format is: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,-1,-1,-1,-1
    """
    print(f"Reading ground truth data from '{csv_path}'...")

    try:
        # Read the CSV with the first 3 rows as a multi-level header,
        # and use the first column ('frame') as the index.
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

        # Name the index and the column levels for easier manipulation
        df.index.name = 'frame'
        df.columns.set_names(['TeamID', 'PlayerID', 'Attributes'], inplace=True)

        # Stack the TeamID and PlayerID levels. This pivots the data from
        # wide to long, moving player information from columns to rows.
        # future_stack=True is added to adopt the new implementation and silence the warning.
        long_df = df.stack(level=['TeamID', 'PlayerID'], future_stack=True)

        # Reset the index to turn 'frame', 'TeamID', and 'PlayerID' into columns
        long_df = long_df.reset_index()

        # Filter out rows where the PlayerID is 'BALL'
        # This prevents the ValueError when trying to convert 'BALL' to an integer.
        long_df = long_df[long_df['PlayerID'] != 'BALL']

        # Check that the reshape produced the necessary columns
        required_cols = ['frame', 'PlayerID', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
        if not all(col in long_df.columns for col in required_cols):
            print("\nError: After reshaping, not all required columns were found.")
            print(f"Required: {required_cols}")
            print(f"Found: {list(long_df.columns)}")
            return

    except Exception as e:
        print(f"\nError processing the CSV file: {e}")
        print("Please ensure the CSV format matches the expected multi-level header structure.")
        return

    print("Successfully read and reshaped CSV. Converting to MOT format...")

    # Create the directory for the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Drop rows with any NaN values, which can occur from the reshape if a
    # player is not present in a particular frame.
    long_df.dropna(subset=['bb_left', 'bb_top', 'bb_width', 'bb_height'], inplace=True)

    with open(output_path, 'w') as f_out:
        for index, row in long_df.iterrows():
            frame = int(row['frame'])
            player_id = int(row['PlayerID'])
            bb_left = float(row['bb_left'])
            bb_top = float(row['bb_top'])
            bb_width = float(row['bb_width'])
            bb_height = float(row['bb_height'])

            # Write in the required MOT format
            # We add '-1' for confidence and x,y,z as they are not used for ground truth
            line = f"{frame},{player_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},-1,-1,-1,-1\n"
            f_out.write(line)

    print(f"Conversion complete. Ground truth saved to '{output_path}'")


if __name__ == "__main__":
    # The path to the CSV file from the dataset
    INPUT_CSV_PATH = "Q4_side_480-510.csv"

    # The desired output path for the converted ground truth file for MOTA, HOTA and IDF1
    OUTPUT_GT_PATH = "data/gt/gt.txt"

    convert_csv_to_mot_format(INPUT_CSV_PATH, OUTPUT_GT_PATH)


