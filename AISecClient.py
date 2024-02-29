from typing import Any, Callable, Optional
import requests
import numpy
from requests.auth import HTTPBasicAuth
import httpx
from huggingface_hub import hf_hub_download
import huggingface_hub
from tqdm import tqdm
import asyncio
import base64
from datetime import datetime


class AISecClient(object):
    def __init__(self, client_id:str, client_secret: str, api_url: str = "https://api.hiddenlayer.ai", retry_count: int = 3):
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_auth = HTTPBasicAuth(client_id, client_secret)
        self.base_url = api_url
        self.token_url = "https://auth.hiddenlayer.ai/oauth2/token?grant_type=client_credentials"
        self.retry_count = retry_count
        self.session = requests.Session()
        self.async_client = httpx.AsyncClient(timeout=None)

    @staticmethod
    def b64str(b: bytes) -> str:
        return base64.b64encode(b).decode()
    
    def _set_token(self) -> str:
        resp = requests.post(self.token_url, auth=self.client_auth)
        if resp.status_code != 200:
            raise Exception("Error: Unable to retrieve token")
        self.token = resp.json()["access_token"]
        self.session.headers = {"Authorization": f"Bearer {self.token}"}
        self.async_client.headers = {"Authorization": f"Bearer {self.token}"}

    def _request_handler(self, meth: Callable, url: str, **kwargs: Any) -> requests.Response:
        resp = None
        for i in range(self.retry_count + 1):
            resp = meth(url, **kwargs)
            if resp.status_code == 401:
                self._set_token()
                continue
            elif resp.status_code < 500:
                break
            sleep(randint(1, i + 1) / 100)  # nosec
        return resp

    async def _async_request_handler(self, meth: Callable, url: str, **kwargs: Any) -> requests.Response:
        resp = None
        for i in range(self.retry_count + 1):
            resp = await meth(url, **kwargs)
            if resp.status_code == 401:
                self._set_token()
                continue
            elif resp.status_code < 500:
                break
            sleep(randint(1, i + 1) / 100)  # nosec
        return resp
    
    def create_sensor(self, name: str, tags: dict = None):
        tags = {} if tags is None else tags
        resp = self._request_handler(self.session.post, f"{self.base_url}/api/v2/sensors/create", json={"plaintext_name": name, "active": True, "tags": tags})
        if not resp.ok:
            raise Exception("Failed to create sensor - demo will not work properly")
        data = resp.json()
        print(f"View model: https://console.hiddenlayer.ai/model-details/{data['model_id']}")
        return data
    
    def get_sensor_by_name_version(self, name: str, version: int):
        resp = self._request_handler(self.session.post, f"{self.base_url}/api/v2/sensors/query", json={"filter":{"plaintext_name": name, "version": version}})
        if not resp.ok:
            raise Exception("Failed to create sensor - demo will not work properly")
        return resp.json()["results"][0]

    def get_or_create_sensor(self, name: str, version: int = 1, tags: dict = None):
        sensor = None
        tags = {} if tags is None else tags
        resp = self._request_handler(self.session.post, f"{self.base_url}/api/v2/sensors/query", json={"filter":{"plaintext_name": name, "version": version}})
        if resp.ok:
            content = resp.json()
            if content["total_count"] >= 1:
                sensor = content["results"][0]

        if sensor is None:
            sensor = self.create_sensor(name, tags=tags)
            
        return sensor    

    async def scan_huggingface_repo(self, repo: str):
        sensors_url = f"{self.base_url}/api/v2/sensors"
        scanning_url = f"{self.base_url}/api/v2/submit/sensors"
        extensions = [".bin", ".h5", ".safetensors"]
        model_info = huggingface_hub.model_info(repo)
        filename = None
        for ext in extensions:
            for s in model_info.siblings:
                if ("/" not in s.rfilename):
                    if s.rfilename.endswith(ext):
                        filename = s.rfilename
                        break
            if filename is not None:
                break

        if filename is None:
            raise Exception(f"Unable to find model to scan in repository: {repo}")

        print(f"Downloading {repo}/{filename} from HuggingFace...")
        filepath_to_model = hf_hub_download(repo_id=repo, filename=filename)
        with open(filepath_to_model, "rb") as fin:
            data = fin.read()
    
        # create sensor
        sensor = self.create_sensor(repo, tags={"env": "demo", "source": "huggingface"})
        sensor_id = sensor["sensor_id"]
        model_id = sensor["model_id"]
    
        # start multipart upload
        print(f"Starting upload for {repo}/{filename}")
        headers = {"X-Content-Length": str(len(data))}
        resp = self.session.post(f"{sensors_url}/{sensor_id}/upload/begin", headers=headers)
        if not resp.ok:
            raise Exception(f"Failed to start multipart upload for {repo}. Demo will not run correctly")
        multipart = resp.json()
    
        # upload parts
        chunk = 8
        upload_id = multipart["upload_id"]
        for i in tqdm(range(0, len(multipart["parts"]), chunk)):
            upload_tasks = []
            group = multipart["parts"][i:i+chunk]
            for p in group:
                part_number = p["part_number"]
                part_data = data[p["start_offset"]:p["end_offset"]]
                t = self._async_request_handler(self.async_client.put, f"{sensors_url}/{sensor_id}/upload/{upload_id}/part/{part_number}", data=part_data, headers=headers)
                upload_tasks.append(t)
            results = await asyncio.gather(*upload_tasks)
    
        # complete multipart upload
        print(f"Completed upload for {repo}/{filename}")
        resp = self.session.post(f"{sensors_url}/{sensor_id}/upload/{upload_id}/complete")
        if not resp.ok:
            raise Exception(f"Failed to complete upload for {repo}. Demo will not run correctly")
    
        # kick off scan for sensor id
        resp = self.session.post(f"{scanning_url}/{sensor_id}/scan")
        if not resp.ok:
            raise Exception("Failed to start model scan. Demo will not run correctly")
        print(f"Scan initiated for {repo}/{filename}. Results will be available shortly at: https://console.hiddenlayer.ai/model-details/{model_id}")
        
    
    def submit_to_mldr(
        self,
        sensor_id: str, 
        requester_id: str,
        input_layer: numpy.ndarray,
        output_layer: numpy.ndarray,
        event_time: str = None,
        predictions: list = None,
        metadata: dict = None
    ) -> None:
        submission_url = f"{self.base_url}/api/v2/submit"
        payload = {
            "sensor_id": sensor_id,
            "requester_id": requester_id,
            "event_time": str(datetime.now().isoformat()) if event_time is None else event_time,
            "input_layer": self.b64str(input_layer.tobytes()),
            "input_layer_shape": list(input_layer.shape),
            "input_layer_dtype": str(input_layer.dtype),
            "output_layer": self.b64str(output_layer.tobytes()),
            "output_layer_shape": list(output_layer.shape),
            "output_layer_dtype": str(output_layer.dtype),
            "predictions": [] if predictions is None else predictions,
            "metadata": {} if metadata is None else metadata,
        }
        resp = self._request_handler(self.session.post, submission_url, json=payload)
        if not resp.ok:
            print("Failed to submit model inference to HiddenLayer")
            print(resp.content)

