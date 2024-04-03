FROM pytorch/torchserve

WORKDIR ./
COPY . .

ENV HOST 0.0.0.0
ENV PORT 8000

RUN pip install -r requirements.txt


EXPOSE ${PORT}
# CMD ["uwsgi", "--ini", "uwsgi.ini"]
CMD ["uvicorn", "app:app", "--host", "${HOST}", "--port", "${PORT}"]