import os
import pinecone


#api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
api_key = "953b2be8-0621-42a1-99db-8480079a9e23"
# find environment next to your API key in the Pinecone console 
#env = "us-east-1-aws"
env = "eu-west4-gcp"

#index_name = "justiz"
index_name = "justiz-openai-full"

pinecone.init(api_key=api_key, environment=env)
#pinecone.whoami()

 # create the index

metadata_config = {
}

pinecone.create_index(
   name = index_name,
   #dimension = 768,  # dimensionality of dense model
   dimension = 1536,
   metric = "dotproduct",  # sparse values supported only for dotproduct
   pod_type = "s1",
   metadata_config=metadata_config
)

index = pinecone.Index(index_name)