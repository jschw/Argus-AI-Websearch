import glob, os
import pdfplumber, re
from tqdm import tqdm
from tabulate import tabulate
from transformers import AutoTokenizer
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from itertools import islice
from sentence_transformers import SentenceTransformer
import torch 


# Parameters:
file_directory = r'N:\ETB\Digitalisierung\12_IPAS_ArtificialIntelligence\Werkstudent\AI-Assistant\Files\**\*.*'

embedding_model = r'C:\Users\UhligMa\python_envs\Models\nomic-ai-nomic-embed-text-v1'
# WARNING:
# For Nomic embedding model data gets specific prompts in line 145 and 197



chunk_size = 500 # length of text chunks
embedding_batch_size = 5 # how many example to process in one batch
max_length = 2048 # max length of input to the embedding model must be >= chunk_size

save_path = f"Chunks_Nomic-{chunk_size}-Token-Embedded" # Where to save the processed dataset with embeddings
# WARNING: If path allready exists this dataset will be loaded

#QDrant Options:
num_results = 5 # How many chunks to return
distance_metric = models.Distance.COSINE
# distance Measurement Options:
# COSINE = "Cosine"
# EUCLID = "Euclid"
# DOT = "Dot"
# MANHATTAN = "Manhattan"

if chunk_size > max_length:
    raise ValueError("Max Length odf Model cannot be smaller than chunk size")


# Load Tokenizer for Text Splitting nad Query Tokenizing
tokenizer = AutoTokenizer.from_pretrained(embedding_model)


# skip preprocessing if dataset exists
# = Frage: Warum laden bzw speichern des Datensatzes? Beim Ausführen der Aktualisierung soll alles neu eingelesen werden.
# ==> Wird nicht benötigt, wenn der Datensatz immer aktualisiert werden soll.
if os.path.isdir(save_path):
    print("Loading Preprocessed Data")
    ds = load_from_disk(save_path)
    print("Loading Model")
    model = SentenceTransformer(
            r'C:\Users\UhligMa\python_envs\Models\nomic-ai-nomic-embed-text-v1', 
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            trust_remote_code=True,
            
    )
    model.max_seq_length = max_length # to prevent OOM Error
    

else:
    print("Preprocessing Files")
    # Load all Files from Directory and Subdirectories:
    files = glob.glob(file_directory, recursive=True)


    # Generated Strucutred Tabels from pdfplumber tables
    def generate_structured_table(table):
        values = [[e for e in tab if e is not None and e !=""] for tab in table]
        columns = max([len(line) for line in values])

        for val in values:
            val.extend([""]*(columns-len(val)))

        return str(tabulate(values,tablefmt="github"))


    texts = []
    page_number = []
    document = []
    for file in tqdm(files,desc="Processing Documents"):
        number = 1 # to start every document with page one
        begin_split_num = number
        # PDF Processing:
        if ".pdf" in file: 
            with pdfplumber.open(file) as pdf:
                length = 0 # initiating chunk length
                final_text = "" # initiating text of the chunk
                
                working_file = file.split('\\')[-1] # get filename to print in progressbar
                for page in tqdm(pdf.pages,desc=f"Working on {working_file}",leave=False):
                    page_text = page.extract_text()
                    page_text = "\n".join([line for line in page_text.split("\n") if "...." not in line]) # remove toc from content 



                    tables = page.extract_tables()
                    table_text=""
                    if tables != None:
                        table_text = "\n".join([generate_structured_table(table) for table in tables])
                        
                    final_text += page_text + "\n" + table_text
                    # = Frage: Entfernen von Kopf + Fußzeile?
                    # ==> Wiederkehrenden Text löschen
                    final_text = final_text.replace("Document name: PR-2616352 I LIMOGES I 4047884 I MultiUse L I FDS\nVersion: V1.0\nProject / Customer: PR-2616352 / Catalent\nMachine number / type: 4047884 / MultiUse","")
                    
                    # = Frage: Was wird hier gemacht?
                    # ==> Mehrere Textfragmente zusammenfügen, wenn sie kleiner als die eingestellte Junksize sind um den Kontext möglichst gut auszunutzen.
                    if len(tokenizer(final_text).input_ids)+length < chunk_size:
                        length += len(tokenizer(final_text).input_ids)
                        begin_split_num = number
                        number +=1 
                    else:
                        texts.append(final_text)
                        page_number.append(begin_split_num) 
                        begin_split_num = number
                        number +=1
                        document.append(file.split("\\")[-1].split(".")[0])
                        final_text = ""
                        length=0
                        
        else:
            with open(file,"r") as f:
                text = f.read()
            texts.append(final_text)
            page_number.append(number)
            number +=1
            document.append(file.split("\\")[-1].split(".")[0])

    # Generate HuggingFace Dataset from Data
    ds = Dataset.from_dict({"text":texts,"document":document,"page_number":page_number})

    print("Loading Model")
    model = SentenceTransformer(
            r'C:\Users\UhligMa\python_envs\Models\nomic-ai-nomic-embed-text-v1', 
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            trust_remote_code=True,
            
    )
    model.max_seq_length = max_length # to prevent OOM Error


    def generate_embeddings(ds, batch_size=embedding_batch_size):
        embeddings = []
        for i in tqdm(range(0, len(ds), batch_size)):
            batch_sentences = ds['text'][i:i+batch_size]
            batch_sentences = [f"search_document: {text}" for text in batch_sentences]
            batch_embeddings = model.encode(batch_sentences)
            embeddings.extend(batch_embeddings)            
        return embeddings

    print("Generating Embeddings")
    embeddings = generate_embeddings(ds)
    ds = ds.add_column("embeddings", embeddings)
    ds = ds.add_column("ids",[i for i in range(len(ds))]) #necessary for Qdrant Store

    # save dataset:
    ds.save_to_disk(save_path)
    print("Dataset saved")

print("Starting Vector Store")
client = QdrantClient(":memory:") # vector store is lost when notebook restarts 
# for permanent store use QdrantClient(path="path/to/db")
# client.delete_collection(collection_name="documents")

client.create_collection(
    collection_name="documents",
    vectors_config=models.VectorParams(
        size=model.get_sentence_embedding_dimension(),
        distance=distance_metric,
    ),
)



def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch

batch_size = 100

for batch in batched(ds, batch_size):
    ids = [point.pop("ids") for point in batch]
    vectors = [point.pop("embeddings") for point in batch]

    client.upsert(
        collection_name="documents",
        points=models.Batch(
            ids=ids,
            vectors=vectors,
            payloads=batch,
        ),
    )


while True:
    query = input("Input Query: ")#"Error during camera calculation"

    hits = client.search(
        collection_name="documents",
        query_vector=model.encode(f"search_query: {query}").tolist(),
        limit=num_results
    )
    for hit in hits:
    # hit.payload = dict{text,document,page_number}
    # score = similarity based on metric
        print(hit.payload["text"], "\nscore:", hit.score)