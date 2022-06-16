from datetime import timedelta


def create_bucket(client=None, bucketName="seeiner-aibox"):
    found = client.bucket_exists(bucketName)
    if not found:
        client.make_bucket(bucketName)
    else:
        print(f"Bucket {bucketName} already exists")


def fput_object(client=None, fileName=None, localPath=None, bucketName="seeiner-aibox"):
    result = client.fput_object(bucketName, fileName, localPath)


def put_object(client=None, fileName=None, data=None, bucketName="seeiner-aibox"):
    result = client.put_object(bucketName, fileName, data, length=-1, part_size=10 * 1024 * 1024)


def get_link(client=None, fileName=None, bucketName="seeiner-aibox"):
    url = client.get_presigned_url("GET", bucketName, fileName, expires=timedelta(days=3650), )
    return url
