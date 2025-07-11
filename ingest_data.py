import json
import os
import weaviate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from utils import preprocess_text

load_dotenv()

class FactCheckDataIngester:
    def __init__(self):
        self.weaviate_url = os.getenv("WEAVIATE_URL", "localhost")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        
        self.client = self._initialize_weaviate_client()
    
    def _initialize_weaviate_client(self):
        """Initialize Weaviate v3 client."""
        try:
            if self.weaviate_api_key and self.weaviate_api_key != "":
                auth_config = weaviate.AuthApiKey(api_key=self.weaviate_api_key)
                client = weaviate.Client(
                    url=f"https://{self.weaviate_url}",
                    auth_client_secret=auth_config
                )
            else:
                client = weaviate.Client(url="http://localhost:8080")
        
            return client
        except Exception as e:
            print(f"Failed to initialize Weaviate client: {e}")
            raise
    
    def create_schema(self, schema_path: str = "weaviate_setup/schema.json"):
        """Create the schema in Weaviate v3."""
        try:
            with open(schema_path, 'r') as f:
                schema_config = json.load(f)
        
            # Check if class exists
            try:
                existing_schema = self.client.schema.get()
                class_names = [cls['class'] for cls in existing_schema.get('classes', [])]
                if schema_config['class'] in class_names:
                    print(f"Class {schema_config['class']} already exists. Skipping creation.")
                    return
            except:
                pass
        
            # Create class
            self.client.schema.create_class(schema_config)
            print(f"Created schema class: {schema_config['class']}")
        
        except Exception as e:
            print(f"Error creating schema: {e}")
            raise
    
    def ingest_fact_checks(self, data_path: str = "data/factbase.json"):
        try:
            print(f"Loading data from {data_path}...")
            with open(data_path, 'r') as f:
                fact_checks = json.load(f)
        
            print(f"Loaded {len(fact_checks)} fact-checks")
        
            print("Preparing texts for embedding...")
            texts = []
            for i, fact_check in enumerate(fact_checks):
                combined_text = f"{fact_check['claim']} {fact_check['explanation']}"
                processed_text = preprocess_text(combined_text)
                texts.append(processed_text)
                print(f"Text {i+1}: {fact_check['claim'][:50]}...")
        
            print(f"Calling OpenAI API to generate embeddings for {len(texts)} texts...")
            print("This may take 2-5 minutes...")
        
            embeddings = self.embeddings.embed_documents(texts)
        
            print(f"Successfully generated {len(embeddings)} embeddings!")
            print("Now inserting into Weaviate...")
        
            # Insert using v3 API
            for i, (fact_check, embedding) in enumerate(zip(fact_checks, embeddings)):
                data_object = {
                    "claim": fact_check["claim"],
                    "verdict": fact_check["verdict"],
                    "explanation": fact_check["explanation"],
                    "source": fact_check["source"],
                    "url": fact_check["url"],
                    "date": fact_check["date"],
                    "category": fact_check["category"],
                    "confidence_score": fact_check["confidence_score"]
                }
                
                # Insert individual object with v3 API
                self.client.data_object.create(
                    data_object=data_object,
                    class_name="FactCheck",
                    vector=embedding
                )
                print(f"Inserted fact-check {i+1}/{len(fact_checks)}")
        
            print(f"Successfully ingested {len(fact_checks)} fact-checks into Weaviate")
        except Exception as e:
            print(f"Error ingesting fact-checks: {e}")
            raise
    
    def close(self):
        pass  # v3 client doesn't need explicit closing

def main():
    try:
        print("Starting fact-check data ingestion...")
        ingester = FactCheckDataIngester()
        
        print("Creating Weaviate schema...")
        ingester.create_schema()
        
        print("Ingesting fact-checks with embeddings...")
        ingester.ingest_fact_checks()
        
        print("Data ingestion completed successfully!")
        
    except Exception as e:
        print(f"Data ingestion failed: {e}")
        return False
    finally:
        if 'ingester' in locals():
            ingester.close()
    
    return True

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()