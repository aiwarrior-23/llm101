import requests
import os
from data_links import links

def download_pdf(url, save_path):
    try:
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        # Send a GET request to the URL
        response = requests.get(url, headers=headers)
        # Check if the request was successful
        if response.status_code == 200:
            # Open a file in binary write mode
            with open(save_path, 'wb') as file:
                # Write the content of the response to the file
                file.write(response.content)
                print(f"PDF downloaded and saved to {save_path}")
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

save_directory = "./data"
for i, pdf_path in enumerate(links):
    save_path = os.path.join(save_directory, f"act_{i+1}.pdf")
    download_pdf(pdf_path, save_path)