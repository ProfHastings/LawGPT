import os
import pinecone


api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
# find environment next to your API key in the Pinecone console 
env = "us-east-1-aws"

index_name = "justiz"

pinecone.init(api_key=api_key, environment=env)
#pinecone.whoami()

 # create the index

metadata_config = {
    "indexed": ["source"]
}

pinecone.create_index(
   name = index_name,
   dimension = 768,  # dimensionality of dense model
   metric = "dotproduct",  # sparse values supported only for dotproduct
   pod_type = "s1",
   #metadata_config=metadata_config  # see explaination above
)

index = pinecone.Index(index_name)