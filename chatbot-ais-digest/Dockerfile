FROM public.ecr.aws/lambda/python:3.10

# Install build essentials
RUN yum update -y && yum groupinstall -y "Development Tools"

COPY requirements.txt ./
RUN pip3 install -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY lambda-main.py ./

CMD ["lambda-main.lambda_handler"]
