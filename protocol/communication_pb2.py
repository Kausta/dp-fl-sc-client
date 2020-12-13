# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protocol/communication.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protocol/communication.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1cprotocol/communication.proto\"=\n\x0fRegisterRequest\x12\x11\n\tclient_id\x18\x01 \x01(\x05\x12\x17\n\x0f\x63lient_data_len\x18\x02 \x01(\x05\"_\n\x10RegisterResponse\x12\x0e\n\x06weight\x18\x01 \x01(\x02\x12\x14\n\x0ctotal_weight\x18\x02 \x01(\x02\x12\x15\n\x05model\x18\x03 \x01(\x0b\x32\x06.Model\x12\x0e\n\x06method\x18\x04 \x01(\t\">\n\x17ShouldContributeRequest\x12\x11\n\tclient_id\x18\x01 \x01(\x05\x12\x10\n\x08last_acc\x18\x02 \x01(\x02\"@\n\x18ShouldContributeResponse\x12\x12\n\ncontribute\x18\x01 \x01(\x08\x12\x10\n\x08\x66inished\x18\x02 \x01(\x08\"?\n\x13\x43ommitUpdateRequest\x12\x11\n\tclient_id\x18\x01 \x01(\x05\x12\x15\n\x05model\x18\x02 \x01(\x0b\x32\x06.Model\"\x15\n\x05Model\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"\t\n\x07VoidMsg\"\x15\n\x03\x41\x63k\x12\x0e\n\x06result\x18\x01 \x01(\x08\x32\xe7\x01\n\x06Server\x12\x39\n\x0eRegisterClient\x12\x10.RegisterRequest\x1a\x11.RegisterResponse\"\x00\x30\x01\x12K\n\x10ShouldContribute\x12\x18.ShouldContributeRequest\x1a\x19.ShouldContributeResponse\"\x00\x30\x01\x12,\n\x0c\x43ommitUpdate\x12\x14.CommitUpdateRequest\x1a\x04.Ack\"\x00\x12\'\n\x0fGetGlobalUpdate\x12\x08.VoidMsg\x1a\x06.Model\"\x00\x30\x01\x62\x06proto3'
)




_REGISTERREQUEST = _descriptor.Descriptor(
  name='RegisterRequest',
  full_name='RegisterRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_id', full_name='RegisterRequest.client_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='client_data_len', full_name='RegisterRequest.client_data_len', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=93,
)


_REGISTERRESPONSE = _descriptor.Descriptor(
  name='RegisterResponse',
  full_name='RegisterResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='weight', full_name='RegisterResponse.weight', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_weight', full_name='RegisterResponse.total_weight', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model', full_name='RegisterResponse.model', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='method', full_name='RegisterResponse.method', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=95,
  serialized_end=190,
)


_SHOULDCONTRIBUTEREQUEST = _descriptor.Descriptor(
  name='ShouldContributeRequest',
  full_name='ShouldContributeRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_id', full_name='ShouldContributeRequest.client_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='last_acc', full_name='ShouldContributeRequest.last_acc', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=192,
  serialized_end=254,
)


_SHOULDCONTRIBUTERESPONSE = _descriptor.Descriptor(
  name='ShouldContributeResponse',
  full_name='ShouldContributeResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='contribute', full_name='ShouldContributeResponse.contribute', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='finished', full_name='ShouldContributeResponse.finished', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=256,
  serialized_end=320,
)


_COMMITUPDATEREQUEST = _descriptor.Descriptor(
  name='CommitUpdateRequest',
  full_name='CommitUpdateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_id', full_name='CommitUpdateRequest.client_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model', full_name='CommitUpdateRequest.model', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=322,
  serialized_end=385,
)


_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='Model.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=387,
  serialized_end=408,
)


_VOIDMSG = _descriptor.Descriptor(
  name='VoidMsg',
  full_name='VoidMsg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=410,
  serialized_end=419,
)


_ACK = _descriptor.Descriptor(
  name='Ack',
  full_name='Ack',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='Ack.result', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=421,
  serialized_end=442,
)

_REGISTERRESPONSE.fields_by_name['model'].message_type = _MODEL
_COMMITUPDATEREQUEST.fields_by_name['model'].message_type = _MODEL
DESCRIPTOR.message_types_by_name['RegisterRequest'] = _REGISTERREQUEST
DESCRIPTOR.message_types_by_name['RegisterResponse'] = _REGISTERRESPONSE
DESCRIPTOR.message_types_by_name['ShouldContributeRequest'] = _SHOULDCONTRIBUTEREQUEST
DESCRIPTOR.message_types_by_name['ShouldContributeResponse'] = _SHOULDCONTRIBUTERESPONSE
DESCRIPTOR.message_types_by_name['CommitUpdateRequest'] = _COMMITUPDATEREQUEST
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['VoidMsg'] = _VOIDMSG
DESCRIPTOR.message_types_by_name['Ack'] = _ACK
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RegisterRequest = _reflection.GeneratedProtocolMessageType('RegisterRequest', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERREQUEST,
  '__module__' : 'protocol.communication_pb2'
  # @@protoc_insertion_point(class_scope:RegisterRequest)
  })
_sym_db.RegisterMessage(RegisterRequest)

RegisterResponse = _reflection.GeneratedProtocolMessageType('RegisterResponse', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERRESPONSE,
  '__module__' : 'protocol.communication_pb2'
  # @@protoc_insertion_point(class_scope:RegisterResponse)
  })
_sym_db.RegisterMessage(RegisterResponse)

ShouldContributeRequest = _reflection.GeneratedProtocolMessageType('ShouldContributeRequest', (_message.Message,), {
  'DESCRIPTOR' : _SHOULDCONTRIBUTEREQUEST,
  '__module__' : 'protocol.communication_pb2'
  # @@protoc_insertion_point(class_scope:ShouldContributeRequest)
  })
_sym_db.RegisterMessage(ShouldContributeRequest)

ShouldContributeResponse = _reflection.GeneratedProtocolMessageType('ShouldContributeResponse', (_message.Message,), {
  'DESCRIPTOR' : _SHOULDCONTRIBUTERESPONSE,
  '__module__' : 'protocol.communication_pb2'
  # @@protoc_insertion_point(class_scope:ShouldContributeResponse)
  })
_sym_db.RegisterMessage(ShouldContributeResponse)

CommitUpdateRequest = _reflection.GeneratedProtocolMessageType('CommitUpdateRequest', (_message.Message,), {
  'DESCRIPTOR' : _COMMITUPDATEREQUEST,
  '__module__' : 'protocol.communication_pb2'
  # @@protoc_insertion_point(class_scope:CommitUpdateRequest)
  })
_sym_db.RegisterMessage(CommitUpdateRequest)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), {
  'DESCRIPTOR' : _MODEL,
  '__module__' : 'protocol.communication_pb2'
  # @@protoc_insertion_point(class_scope:Model)
  })
_sym_db.RegisterMessage(Model)

VoidMsg = _reflection.GeneratedProtocolMessageType('VoidMsg', (_message.Message,), {
  'DESCRIPTOR' : _VOIDMSG,
  '__module__' : 'protocol.communication_pb2'
  # @@protoc_insertion_point(class_scope:VoidMsg)
  })
_sym_db.RegisterMessage(VoidMsg)

Ack = _reflection.GeneratedProtocolMessageType('Ack', (_message.Message,), {
  'DESCRIPTOR' : _ACK,
  '__module__' : 'protocol.communication_pb2'
  # @@protoc_insertion_point(class_scope:Ack)
  })
_sym_db.RegisterMessage(Ack)



_SERVER = _descriptor.ServiceDescriptor(
  name='Server',
  full_name='Server',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=445,
  serialized_end=676,
  methods=[
  _descriptor.MethodDescriptor(
    name='RegisterClient',
    full_name='Server.RegisterClient',
    index=0,
    containing_service=None,
    input_type=_REGISTERREQUEST,
    output_type=_REGISTERRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ShouldContribute',
    full_name='Server.ShouldContribute',
    index=1,
    containing_service=None,
    input_type=_SHOULDCONTRIBUTEREQUEST,
    output_type=_SHOULDCONTRIBUTERESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='CommitUpdate',
    full_name='Server.CommitUpdate',
    index=2,
    containing_service=None,
    input_type=_COMMITUPDATEREQUEST,
    output_type=_ACK,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetGlobalUpdate',
    full_name='Server.GetGlobalUpdate',
    index=3,
    containing_service=None,
    input_type=_VOIDMSG,
    output_type=_MODEL,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SERVER)

DESCRIPTOR.services_by_name['Server'] = _SERVER

# @@protoc_insertion_point(module_scope)
