import json
import csv
import os

def convert_json_to_csv(json_file_path, csv_file_path):
    """
    Convert JSON news data to CSV format.
    
    Args:
        json_file_path (str): Path to the input JSON file
        csv_file_path (str): Path to the output CSV file
    """
    try:
        # Read the JSONL file (one JSON object per line)
        data = []
        with open(json_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    item = json.loads(line.strip())
                    data.append(item)
        
        if not data:
            print("Error: No data found in JSON file")
            return False
        
        print(f"Successfully loaded {len(data)} items from the JSON file")
        
        # Get all field names from the data
        # Use the field order we defined previously
        field_names = ["headline", "authors", "short_description", "date", "link", "category", "content"]
        
        # Ensure all field names are included
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        # Add any fields that weren't in our predefined order
        for field in all_fields:
            if field not in field_names:
                field_names.append(field)
        
        # Write the CSV file
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Successfully converted JSON to CSV and saved to {csv_file_path}")
        return True
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Input and output file paths
    json_file_path = "input.json"
    csv_file_path = "final_data.csv"
    
    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File '{json_file_path}' does not exist.")
        return
    
    # Convert JSON to CSV
    convert_json_to_csv(json_file_path, csv_file_path)

if __name__ == "__main__":
    main()
