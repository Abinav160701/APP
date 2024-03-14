from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import os

def get_similar_skus(query_features):
    AWS_ACCESS_KEY_ID = 'AKIA55E32CSUIWHBYM2Y'
    AWS_SECRET_ACCESS_KEY = '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'
    AWS_DEFAULT_REGION = 'eu-west-1'

    service = 'es'  # AWS OpenSearch service code
    awsauth = AWS4Auth(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, service)

    # OpenSearch domain details
    host = 'vpc-image-search-psjcvq3bkrrl22bzgjz6dwcw6m.eu-west-1.es.amazonaws.com'  # Replace with your OpenSearch domain endpoint
    port = 443  # Use port 443 for HTTPS
    use_ssl = True

    # Initialize OpenSearch client
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=awsauth,
        use_ssl=use_ssl,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    search_query = {
        "size": 10,  # Number of results to return
        "query": {
            "knn": {
                "image-vector": {
                    "vector": query_features[0],
                    "k": 10  # Adjust k value based on how many nearest neighbors you want
                }
            }
        }
    }

    response = client.search(index="image_search_test", body=search_query)

    # Extract similar SKUs from the search response
    similar_skus = [hit["_source"]["sku"] for hit in response["hits"]["hits"]]

    return similar_skus
