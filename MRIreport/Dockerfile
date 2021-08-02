FROM python:3.9-slim-buster

LABEL IMAGE="MRIReport"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="True"

RUN apt-get update
RUN apt-get -y install libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

RUN pip install numpy==1.21.0
RUN pip install pandas==1.3.0
RUN pip install seaborn==0.11.1 
RUN pip install plotly==5.1.0
RUN pip install pygam==0.8.0
RUN pip install scipy==1.7.0
RUN pip install ipython==7.12.0
RUN pip install pynrrd==0.4.2
RUN pip install matplotlib==3.4.2
RUN pip install glob2
RUN pip install weasyprint
RUN pip install kaleido

RUN mkdir /src 
COPY files/MRIReport.py /src
COPY files/reportdriver.py /src

RUN mkdir /Ref_CSV
COPY Ref_CSV/ /Ref_CSV

RUN mkdir /data
COPY data/ /data

CMD ["python3","-u","src/reportdriver.py"]
