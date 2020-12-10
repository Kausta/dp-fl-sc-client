#!/bin/bash

# Compile for python
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./communication/communication.proto