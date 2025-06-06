import boto3
import os
from dotenv import load_dotenv
import time


load_dotenv()

# AWS 자격증명과 리전 설정
aws_access_key_id = os.getenv("S3_ACCESS_KEY")
aws_secret_access_key = os.getenv("S3_SECRET_KEY")
region_name = "ap-southeast-2"  # 시드니 리전

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# bucket_name = "your-bucket-name"
# object_name = "folder/example.txt"    # S3에 저장될 객체 경로 및 파일 이름
# file_path = "./local_example.txt"     # 업로드할 파일 경로

# # S3에 업로드 (public-read 권한 부여)
# s3.upload_file(
#     file_path, bucket_name, object_name,
#     ExtraArgs={'ACL': 'public-read'}   # 중요: public 읽기 권한 부여
# )
#
# # S3 퍼블릭 URL 생성
# url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
#
# print('업로드 완료!')
# print(f'파일 URL: {url}')



def upload_obj(object_name, file_obj):
    bucket_name = "songil-s3"
    timestamp_seconds_str = str(int(time.time()))

    s3.upload_file(file_obj, bucket_name, timestamp_seconds_str+object_name)
    return f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{timestamp_seconds_str+object_name}"