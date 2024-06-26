from argus_src.core import ArgusWebsearch
import time

# user_prompt = "What's the world's tallest building in 2023?"

# user_prompt = "What is the reason why some stars are flickering?"

# user_prompt = "Is the Bielefeld conspiracy real?"

# user_prompt = "What was the tallest building in the world in the year 2019?"

# user_prompt = "The Apple Watch is not visible on my new iPhone and I cannot connect it. How do I configure it for the new phone?"

# user_prompt = "My coffee machine has stopped working, no water comes out. Maybe it needs cleaning? But I don't know how to do that."

# user_prompt = "Please make a research about the CO2 footprint of battery electric vehicles and compare with an equivalent vehicle with a modern Diesel engine."

user_prompt = "Is it true that an apple a day keeps the doctor away? Give me the reason for the answer."

user_prompt = "How to grow Cannabis? Name the steps how to grow cannabis on the balcony and recommend three suitable varieties."


search_engine = ArgusWebsearch()

first_cycle = True

while True:
    inp = input("Input: ")

    if inp == "quit": break

    if inp:
        user_prompt = inp
        
        if first_cycle:
            text_out, tokens_used = search_engine.run_full_research(input_prompt=user_prompt)

            print("AI response:")
            print(text_out)
            print("\nURLs:")

            # Print unique source URLs
            urls_added = []
            url_numbers = []

            src_num = 1
            for index, doc in search_engine.rag_context.iterrows():

                # Add URL only once
                if doc['URL'] in urls_added:
                    # Source is alread added
                    # Get index num
                    src_idx = urls_added.index(doc['URL'])

                    # Add source number to array
                    url_numbers[src_idx].append(str(src_num))

                else:
                    # Source is not present in array
                    urls_added.append(doc['URL'])
                    url_numbers.append([str(src_num)])
                
                src_num += 1

            # Print URLs and source numbers to output
            i = 0
            for url_src in urls_added:
                src_numbers = ', '.join(url_numbers[i])

                print(f"{src_numbers}: {url_src}")

                i += 1

            print(f"\n--> Total tokens used for research: {tokens_used}\n")

            first_cycle = False

        else:
            text_out, tokens_used = search_engine.append_and_run(input_prompt=user_prompt)

            print("AI response:")
            print(text_out)
            print(f"\n--> Tokens used for inference: {tokens_used}\n")

    time.sleep(0.01)

