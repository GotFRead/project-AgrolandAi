import asyncio
import datetime
import logging
import json
from enum import Enum
import analyze_module
from flask import Flask
from flask import request
from flask import abort
import base64


class Type_Deployment(str, Enum):
    SAFE = 'safe_deploy'
    EASY = 'easy_integration'


class SeparatorBytes(bytes, Enum):
    START = b'\xfa\xfa\xfa'
    END = b'\xff\xff\xff'


def create_logger(logger_name):
    logger = logging.getLogger(str(logger_name))
    logger.setLevel(logging.INFO)

    file_logger = logging.FileHandler('service_log.txt')

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_logger.setFormatter(formatter)

    logger.addHandler(file_logger)
    return logger


logger = create_logger("service_main_thread")

response_template = {
    'content': '',
    'description':''
}


class Service:
    """
    Template recv message json format
        example:
        {
        'content': 'any_content',
        'parameters': {
            'parms1': 1,
            'parms2': 2
        }
    }
    """

    def __init__(self, name_instance='test', ip='127.0.0.1', port=45000, type_deployment=Type_Deployment.SAFE, route="/") -> None:
        logger.info('Service INIT!')
        if type_deployment is Type_Deployment.EASY:
            self.main_easy_deploy(route)

        elif type_deployment is Type_Deployment.SAFE:
            self.name = name_instance
            self.ip = ip
            self.port = port
            self.time_start = datetime.datetime.now()
            self.event_loop = asyncio.get_event_loop()
            self.queue = asyncio.Queue()

            asyncio.run(self.main_safe_deploy())

        logger.info('Server START!')

    def get_start_time(self):
        return self.time_start

    def get_name_instance(self):
        return self.name

    async def message_handler(self):
        while True:
            _, payload = self.queue.get()
            payload = await self.decode_payload(payload)
            await self.init_solution(*self.get_content(payload))

    async def separation_payload(self):
        payload = self.recv_message.split(SeparatorBytes.START)[-1]
        payload, self.recv_message = payload.split(SeparatorBytes.END)
        logger.info(payload)
        return payload

    async def decode_payload(self, payload):
        return payload.decode()

    def get_content(self, payload):
        deserial = json.loads(payload)
        content = deserial['content']
        params = deserial['parameters']
        logger.info(content)
        return content, params

    def init_solution(self, content, params):
        return analyze_module.solutionFactory(content, params).execution()

    async def main_safe_deploy(self):
        server = await asyncio.start_server(
            self.handler,
            self.ip,
            self.port
        )

        async with server:
            await server.serve_forever()

    async def handler_safe_deploy(self, reader, writer):
        self.recv_message = await reader.read(100)
        if SeparatorBytes.START in self.recv_message and SeparatorBytes.END in self.recv_message:
            payload = await self.separation_payload()
            self.queue.put_nowait(payload)

    def get_content_easy_deploy(self, payload):
        content = base64.b64decode(payload['content'])
        params = payload['parameters']
        logger.info(content)
        return content, params

    def main_easy_deploy(self, route):
        app = Flask("Service")

        @app.route(route, methods=['POST'])
        def analyze():
            if not request.json or 'content' not in request.json or 'parameters' not in request.json:
                abort(400)
            response = response_template
            content = self.init_solution(*self.get_content_easy_deploy(request.json))
            response['content'] = content.decode('ascii')
            return response
        app.run(debug=False)


if __name__ == '__main__':
    service = Service(type_deployment=Type_Deployment.EASY,
                      route='/analyze')
