import psycopg2
import boto3
import base64
import json
from botocore.exceptions import ClientError



def get_secret(secret_name):

    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            return json.loads(get_secret_value_response['SecretString'])
        else:
            return base64.b64decode(get_secret_value_response['SecretBinary'])
        

def almacena_vector(nombre, descripcion, vector):
    secret = get_secret('loginexperience/database')
    conn = psycopg2.connect(
        dbname=secret['dbInstanceIdentifier'],
        user=secret['username'],
        password=secret['password'],
        host=secret['host'],
        port=secret['port']
    )

    # Crea un cursor para ejecutar consultas
    cur = conn.cursor()


    # Inserta el vector en la tabla
    query = "INSERT INTO vector_data (nombre, descripcion, vector) VALUES (%s, %s, %s);"
    cur.execute(query, (nombre, descripcion, vector))

    # Confirma la transacción y cierra la conexión
    conn.commit()
    cur.close()
    conn.close()


def busqueda_similitud(query_vector):

    secret = get_secret('loginexperience/database')
    conn = psycopg2.connect(
        dbname=secret['dbInstanceIdentifier'],
        user=secret['username'],
        password=secret['password'],
        host=secret['host'],
        port=secret['port']
    )

    # Crea un cursor para ejecutar consultas
    cur = conn.cursor()

    # Ejecuta la consulta que utiliza sml_cosine
    query = '''
    SELECT id, nombre, descripcion, vector, cosine_similarity(vector, %s) AS similarity
    FROM vector_data
    ORDER BY similarity DESC
    LIMIT 10;
    '''
    cur.execute(query, (query_vector,))

    # Recupera los resultados
    results = cur.fetchall()

    # Imprime los resultados
    for result in results:
        id, nombre, descripcion, vector, similarity = result
        print(f"ID: {id}, Nombre: {nombre}, Descripcion: {descripcion}, Similarity: {similarity}")

    # Cierra la conexión
    cur.close()
    conn.close()
