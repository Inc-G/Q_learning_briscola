��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8��
�
'my_model_dense_save_5/dense_9254/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'my_model_dense_save_5/dense_9254/kernel
�
;my_model_dense_save_5/dense_9254/kernel/Read/ReadVariableOpReadVariableOp'my_model_dense_save_5/dense_9254/kernel* 
_output_shapes
:
��*
dtype0
�
%my_model_dense_save_5/dense_9254/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%my_model_dense_save_5/dense_9254/bias
�
9my_model_dense_save_5/dense_9254/bias/Read/ReadVariableOpReadVariableOp%my_model_dense_save_5/dense_9254/bias*
_output_shapes	
:�*
dtype0
�
'my_model_dense_save_5/dense_9255/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'my_model_dense_save_5/dense_9255/kernel
�
;my_model_dense_save_5/dense_9255/kernel/Read/ReadVariableOpReadVariableOp'my_model_dense_save_5/dense_9255/kernel* 
_output_shapes
:
��*
dtype0
�
%my_model_dense_save_5/dense_9255/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%my_model_dense_save_5/dense_9255/bias
�
9my_model_dense_save_5/dense_9255/bias/Read/ReadVariableOpReadVariableOp%my_model_dense_save_5/dense_9255/bias*
_output_shapes	
:�*
dtype0
�
'my_model_dense_save_5/dense_9256/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*8
shared_name)'my_model_dense_save_5/dense_9256/kernel
�
;my_model_dense_save_5/dense_9256/kernel/Read/ReadVariableOpReadVariableOp'my_model_dense_save_5/dense_9256/kernel*
_output_shapes
:	�*
dtype0
�
%my_model_dense_save_5/dense_9256/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%my_model_dense_save_5/dense_9256/bias
�
9my_model_dense_save_5/dense_9256/bias/Read/ReadVariableOpReadVariableOp%my_model_dense_save_5/dense_9256/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	my_layers
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures

	0

1
2
 
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
�
layer_regularization_losses
	variables
non_trainable_variables
layer_metrics
trainable_variables

layers
metrics
regularization_losses
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
ca
VARIABLE_VALUE'my_model_dense_save_5/dense_9254/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%my_model_dense_save_5/dense_9254/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'my_model_dense_save_5/dense_9255/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%my_model_dense_save_5/dense_9255/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'my_model_dense_save_5/dense_9256/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%my_model_dense_save_5/dense_9256/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

	0

1
2
 

0
1

0
1
 
�
	variables
#non_trainable_variables
$layer_metrics
trainable_variables

%layers
regularization_losses
&metrics
'layer_regularization_losses

0
1

0
1
 
�
	variables
(non_trainable_variables
)layer_metrics
trainable_variables

*layers
regularization_losses
+metrics
,layer_regularization_losses

0
1

0
1
 
�
	variables
-non_trainable_variables
.layer_metrics
 trainable_variables

/layers
!regularization_losses
0metrics
1layer_regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_1Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1'my_model_dense_save_5/dense_9254/kernel%my_model_dense_save_5/dense_9254/bias'my_model_dense_save_5/dense_9255/kernel%my_model_dense_save_5/dense_9255/bias'my_model_dense_save_5/dense_9256/kernel%my_model_dense_save_5/dense_9256/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *1
f,R*
(__inference_signature_wrapper_1217095104
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename;my_model_dense_save_5/dense_9254/kernel/Read/ReadVariableOp9my_model_dense_save_5/dense_9254/bias/Read/ReadVariableOp;my_model_dense_save_5/dense_9255/kernel/Read/ReadVariableOp9my_model_dense_save_5/dense_9255/bias/Read/ReadVariableOp;my_model_dense_save_5/dense_9256/kernel/Read/ReadVariableOp9my_model_dense_save_5/dense_9256/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_save_1217095306
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename'my_model_dense_save_5/dense_9254/kernel%my_model_dense_save_5/dense_9254/bias'my_model_dense_save_5/dense_9255/kernel%my_model_dense_save_5/dense_9255/bias'my_model_dense_save_5/dense_9256/kernel%my_model_dense_save_5/dense_9256/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference__traced_restore_1217095334��
�	
�
:__inference_my_model_dense_save_5_layer_call_fn_1217095154	
state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_12170950662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*C
_input_shapes2
0:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:����������

_user_specified_namestate
�(
�
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_1217095185
input_1-
)dense_9254_matmul_readvariableop_resource.
*dense_9254_biasadd_readvariableop_resource-
)dense_9255_matmul_readvariableop_resource.
*dense_9255_biasadd_readvariableop_resource-
)dense_9256_matmul_readvariableop_resource.
*dense_9256_biasadd_readvariableop_resource

identity_1

identity_2��!dense_9254/BiasAdd/ReadVariableOp� dense_9254/MatMul/ReadVariableOp�!dense_9255/BiasAdd/ReadVariableOp� dense_9255/MatMul/ReadVariableOp�!dense_9256/BiasAdd/ReadVariableOp� dense_9256/MatMul/ReadVariableOp`
IdentityIdentityinput_1*
T0*,
_output_shapes
:����������2

Identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
 dense_9254/MatMul/ReadVariableOpReadVariableOp)dense_9254_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 dense_9254/MatMul/ReadVariableOp�
dense_9254/MatMulMatMulstrided_slice:output:0(dense_9254/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9254/MatMul�
!dense_9254/BiasAdd/ReadVariableOpReadVariableOp*dense_9254_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!dense_9254/BiasAdd/ReadVariableOp�
dense_9254/BiasAddBiasAdddense_9254/MatMul:product:0)dense_9254/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9254/BiasAddz
dense_9254/TanhTanhdense_9254/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9254/Tanh�
 dense_9255/MatMul/ReadVariableOpReadVariableOp)dense_9255_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 dense_9255/MatMul/ReadVariableOp�
dense_9255/MatMulMatMuldense_9254/Tanh:y:0(dense_9255/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9255/MatMul�
!dense_9255/BiasAdd/ReadVariableOpReadVariableOp*dense_9255_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!dense_9255/BiasAdd/ReadVariableOp�
dense_9255/BiasAddBiasAdddense_9255/MatMul:product:0)dense_9255/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9255/BiasAddz
dense_9255/TanhTanhdense_9255/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9255/Tanh�
 dense_9256/MatMul/ReadVariableOpReadVariableOp)dense_9256_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 dense_9256/MatMul/ReadVariableOp�
dense_9256/MatMulMatMuldense_9255/Tanh:y:0(dense_9256/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_9256/MatMul�
!dense_9256/BiasAdd/ReadVariableOpReadVariableOp*dense_9256_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_9256/BiasAdd/ReadVariableOp�
dense_9256/BiasAddBiasAdddense_9256/MatMul:product:0)dense_9256/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_9256/BiasAdd�
dense_9256/SigmoidSigmoiddense_9256/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_9256/Sigmoid�

Identity_1Identitydense_9256/Sigmoid:y:0"^dense_9254/BiasAdd/ReadVariableOp!^dense_9254/MatMul/ReadVariableOp"^dense_9255/BiasAdd/ReadVariableOp!^dense_9255/MatMul/ReadVariableOp"^dense_9256/BiasAdd/ReadVariableOp!^dense_9256/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identitydense_9256/Sigmoid:y:0"^dense_9254/BiasAdd/ReadVariableOp!^dense_9254/MatMul/ReadVariableOp"^dense_9255/BiasAdd/ReadVariableOp!^dense_9255/MatMul/ReadVariableOp"^dense_9256/BiasAdd/ReadVariableOp!^dense_9256/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*C
_input_shapes2
0:����������::::::2F
!dense_9254/BiasAdd/ReadVariableOp!dense_9254/BiasAdd/ReadVariableOp2D
 dense_9254/MatMul/ReadVariableOp dense_9254/MatMul/ReadVariableOp2F
!dense_9255/BiasAdd/ReadVariableOp!dense_9255/BiasAdd/ReadVariableOp2D
 dense_9255/MatMul/ReadVariableOp dense_9255/MatMul/ReadVariableOp2F
!dense_9256/BiasAdd/ReadVariableOp!dense_9256/BiasAdd/ReadVariableOp2D
 dense_9256/MatMul/ReadVariableOp dense_9256/MatMul/ReadVariableOp:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�(
�
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_1217095135	
state-
)dense_9254_matmul_readvariableop_resource.
*dense_9254_biasadd_readvariableop_resource-
)dense_9255_matmul_readvariableop_resource.
*dense_9255_biasadd_readvariableop_resource-
)dense_9256_matmul_readvariableop_resource.
*dense_9256_biasadd_readvariableop_resource

identity_1

identity_2��!dense_9254/BiasAdd/ReadVariableOp� dense_9254/MatMul/ReadVariableOp�!dense_9255/BiasAdd/ReadVariableOp� dense_9255/MatMul/ReadVariableOp�!dense_9256/BiasAdd/ReadVariableOp� dense_9256/MatMul/ReadVariableOp^
IdentityIdentitystate*
T0*,
_output_shapes
:����������2

Identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSlicestatestrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
 dense_9254/MatMul/ReadVariableOpReadVariableOp)dense_9254_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 dense_9254/MatMul/ReadVariableOp�
dense_9254/MatMulMatMulstrided_slice:output:0(dense_9254/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9254/MatMul�
!dense_9254/BiasAdd/ReadVariableOpReadVariableOp*dense_9254_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!dense_9254/BiasAdd/ReadVariableOp�
dense_9254/BiasAddBiasAdddense_9254/MatMul:product:0)dense_9254/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9254/BiasAddz
dense_9254/TanhTanhdense_9254/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9254/Tanh�
 dense_9255/MatMul/ReadVariableOpReadVariableOp)dense_9255_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 dense_9255/MatMul/ReadVariableOp�
dense_9255/MatMulMatMuldense_9254/Tanh:y:0(dense_9255/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9255/MatMul�
!dense_9255/BiasAdd/ReadVariableOpReadVariableOp*dense_9255_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!dense_9255/BiasAdd/ReadVariableOp�
dense_9255/BiasAddBiasAdddense_9255/MatMul:product:0)dense_9255/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9255/BiasAddz
dense_9255/TanhTanhdense_9255/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9255/Tanh�
 dense_9256/MatMul/ReadVariableOpReadVariableOp)dense_9256_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 dense_9256/MatMul/ReadVariableOp�
dense_9256/MatMulMatMuldense_9255/Tanh:y:0(dense_9256/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_9256/MatMul�
!dense_9256/BiasAdd/ReadVariableOpReadVariableOp*dense_9256_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_9256/BiasAdd/ReadVariableOp�
dense_9256/BiasAddBiasAdddense_9256/MatMul:product:0)dense_9256/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_9256/BiasAdd�
dense_9256/SigmoidSigmoiddense_9256/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_9256/Sigmoid�

Identity_1Identitydense_9256/Sigmoid:y:0"^dense_9254/BiasAdd/ReadVariableOp!^dense_9254/MatMul/ReadVariableOp"^dense_9255/BiasAdd/ReadVariableOp!^dense_9255/MatMul/ReadVariableOp"^dense_9256/BiasAdd/ReadVariableOp!^dense_9256/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identitydense_9256/Sigmoid:y:0"^dense_9254/BiasAdd/ReadVariableOp!^dense_9254/MatMul/ReadVariableOp"^dense_9255/BiasAdd/ReadVariableOp!^dense_9255/MatMul/ReadVariableOp"^dense_9256/BiasAdd/ReadVariableOp!^dense_9256/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*C
_input_shapes2
0:����������::::::2F
!dense_9254/BiasAdd/ReadVariableOp!dense_9254/BiasAdd/ReadVariableOp2D
 dense_9254/MatMul/ReadVariableOp dense_9254/MatMul/ReadVariableOp2F
!dense_9255/BiasAdd/ReadVariableOp!dense_9255/BiasAdd/ReadVariableOp2D
 dense_9255/MatMul/ReadVariableOp dense_9255/MatMul/ReadVariableOp2F
!dense_9256/BiasAdd/ReadVariableOp!dense_9256/BiasAdd/ReadVariableOp2D
 dense_9256/MatMul/ReadVariableOp dense_9256/MatMul/ReadVariableOp:S O
,
_output_shapes
:����������

_user_specified_namestate
�	
�
J__inference_dense_9254_layer_call_and_return_conditional_losses_1217094966

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
J__inference_dense_9254_layer_call_and_return_conditional_losses_1217095215

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_dense_9254_layer_call_fn_1217095224

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_9254_layer_call_and_return_conditional_losses_12170949662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference__traced_save_1217095306
file_prefixF
Bsavev2_my_model_dense_save_5_dense_9254_kernel_read_readvariableopD
@savev2_my_model_dense_save_5_dense_9254_bias_read_readvariableopF
Bsavev2_my_model_dense_save_5_dense_9255_kernel_read_readvariableopD
@savev2_my_model_dense_save_5_dense_9255_bias_read_readvariableopF
Bsavev2_my_model_dense_save_5_dense_9256_kernel_read_readvariableopD
@savev2_my_model_dense_save_5_dense_9256_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_my_model_dense_save_5_dense_9254_kernel_read_readvariableop@savev2_my_model_dense_save_5_dense_9254_bias_read_readvariableopBsavev2_my_model_dense_save_5_dense_9255_kernel_read_readvariableop@savev2_my_model_dense_save_5_dense_9255_bias_read_readvariableopBsavev2_my_model_dense_save_5_dense_9256_kernel_read_readvariableop@savev2_my_model_dense_save_5_dense_9256_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*N
_input_shapes=
;: :
��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�	
�
(__inference_signature_wrapper_1217095104
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference__wrapped_model_12170949462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*C
_input_shapes2
0:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
/__inference_dense_9256_layer_call_fn_1217095264

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_9256_layer_call_and_return_conditional_losses_12170950202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
J__inference_dense_9256_layer_call_and_return_conditional_losses_1217095255

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
:__inference_my_model_dense_save_5_layer_call_fn_1217095204
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_12170950662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*C
_input_shapes2
0:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�9
�
%__inference__wrapped_model_1217094946
input_1C
?my_model_dense_save_5_dense_9254_matmul_readvariableop_resourceD
@my_model_dense_save_5_dense_9254_biasadd_readvariableop_resourceC
?my_model_dense_save_5_dense_9255_matmul_readvariableop_resourceD
@my_model_dense_save_5_dense_9255_biasadd_readvariableop_resourceC
?my_model_dense_save_5_dense_9256_matmul_readvariableop_resourceD
@my_model_dense_save_5_dense_9256_biasadd_readvariableop_resource
identity

identity_1��7my_model_dense_save_5/dense_9254/BiasAdd/ReadVariableOp�6my_model_dense_save_5/dense_9254/MatMul/ReadVariableOp�7my_model_dense_save_5/dense_9255/BiasAdd/ReadVariableOp�6my_model_dense_save_5/dense_9255/MatMul/ReadVariableOp�7my_model_dense_save_5/dense_9256/BiasAdd/ReadVariableOp�6my_model_dense_save_5/dense_9256/MatMul/ReadVariableOp�
my_model_dense_save_5/IdentityIdentityinput_1*
T0*,
_output_shapes
:����������2 
my_model_dense_save_5/Identity�
)my_model_dense_save_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    2+
)my_model_dense_save_5/strided_slice/stack�
+my_model_dense_save_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2-
+my_model_dense_save_5/strided_slice/stack_1�
+my_model_dense_save_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+my_model_dense_save_5/strided_slice/stack_2�
#my_model_dense_save_5/strided_sliceStridedSliceinput_12my_model_dense_save_5/strided_slice/stack:output:04my_model_dense_save_5/strided_slice/stack_1:output:04my_model_dense_save_5/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2%
#my_model_dense_save_5/strided_slice�
6my_model_dense_save_5/dense_9254/MatMul/ReadVariableOpReadVariableOp?my_model_dense_save_5_dense_9254_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype028
6my_model_dense_save_5/dense_9254/MatMul/ReadVariableOp�
'my_model_dense_save_5/dense_9254/MatMulMatMul,my_model_dense_save_5/strided_slice:output:0>my_model_dense_save_5/dense_9254/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'my_model_dense_save_5/dense_9254/MatMul�
7my_model_dense_save_5/dense_9254/BiasAdd/ReadVariableOpReadVariableOp@my_model_dense_save_5_dense_9254_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype029
7my_model_dense_save_5/dense_9254/BiasAdd/ReadVariableOp�
(my_model_dense_save_5/dense_9254/BiasAddBiasAdd1my_model_dense_save_5/dense_9254/MatMul:product:0?my_model_dense_save_5/dense_9254/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(my_model_dense_save_5/dense_9254/BiasAdd�
%my_model_dense_save_5/dense_9254/TanhTanh1my_model_dense_save_5/dense_9254/BiasAdd:output:0*
T0*(
_output_shapes
:����������2'
%my_model_dense_save_5/dense_9254/Tanh�
6my_model_dense_save_5/dense_9255/MatMul/ReadVariableOpReadVariableOp?my_model_dense_save_5_dense_9255_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype028
6my_model_dense_save_5/dense_9255/MatMul/ReadVariableOp�
'my_model_dense_save_5/dense_9255/MatMulMatMul)my_model_dense_save_5/dense_9254/Tanh:y:0>my_model_dense_save_5/dense_9255/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'my_model_dense_save_5/dense_9255/MatMul�
7my_model_dense_save_5/dense_9255/BiasAdd/ReadVariableOpReadVariableOp@my_model_dense_save_5_dense_9255_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype029
7my_model_dense_save_5/dense_9255/BiasAdd/ReadVariableOp�
(my_model_dense_save_5/dense_9255/BiasAddBiasAdd1my_model_dense_save_5/dense_9255/MatMul:product:0?my_model_dense_save_5/dense_9255/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(my_model_dense_save_5/dense_9255/BiasAdd�
%my_model_dense_save_5/dense_9255/TanhTanh1my_model_dense_save_5/dense_9255/BiasAdd:output:0*
T0*(
_output_shapes
:����������2'
%my_model_dense_save_5/dense_9255/Tanh�
6my_model_dense_save_5/dense_9256/MatMul/ReadVariableOpReadVariableOp?my_model_dense_save_5_dense_9256_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype028
6my_model_dense_save_5/dense_9256/MatMul/ReadVariableOp�
'my_model_dense_save_5/dense_9256/MatMulMatMul)my_model_dense_save_5/dense_9255/Tanh:y:0>my_model_dense_save_5/dense_9256/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2)
'my_model_dense_save_5/dense_9256/MatMul�
7my_model_dense_save_5/dense_9256/BiasAdd/ReadVariableOpReadVariableOp@my_model_dense_save_5_dense_9256_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7my_model_dense_save_5/dense_9256/BiasAdd/ReadVariableOp�
(my_model_dense_save_5/dense_9256/BiasAddBiasAdd1my_model_dense_save_5/dense_9256/MatMul:product:0?my_model_dense_save_5/dense_9256/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(my_model_dense_save_5/dense_9256/BiasAdd�
(my_model_dense_save_5/dense_9256/SigmoidSigmoid1my_model_dense_save_5/dense_9256/BiasAdd:output:0*
T0*'
_output_shapes
:���������2*
(my_model_dense_save_5/dense_9256/Sigmoid�
IdentityIdentity,my_model_dense_save_5/dense_9256/Sigmoid:y:08^my_model_dense_save_5/dense_9254/BiasAdd/ReadVariableOp7^my_model_dense_save_5/dense_9254/MatMul/ReadVariableOp8^my_model_dense_save_5/dense_9255/BiasAdd/ReadVariableOp7^my_model_dense_save_5/dense_9255/MatMul/ReadVariableOp8^my_model_dense_save_5/dense_9256/BiasAdd/ReadVariableOp7^my_model_dense_save_5/dense_9256/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity,my_model_dense_save_5/dense_9256/Sigmoid:y:08^my_model_dense_save_5/dense_9254/BiasAdd/ReadVariableOp7^my_model_dense_save_5/dense_9254/MatMul/ReadVariableOp8^my_model_dense_save_5/dense_9255/BiasAdd/ReadVariableOp7^my_model_dense_save_5/dense_9255/MatMul/ReadVariableOp8^my_model_dense_save_5/dense_9256/BiasAdd/ReadVariableOp7^my_model_dense_save_5/dense_9256/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*C
_input_shapes2
0:����������::::::2r
7my_model_dense_save_5/dense_9254/BiasAdd/ReadVariableOp7my_model_dense_save_5/dense_9254/BiasAdd/ReadVariableOp2p
6my_model_dense_save_5/dense_9254/MatMul/ReadVariableOp6my_model_dense_save_5/dense_9254/MatMul/ReadVariableOp2r
7my_model_dense_save_5/dense_9255/BiasAdd/ReadVariableOp7my_model_dense_save_5/dense_9255/BiasAdd/ReadVariableOp2p
6my_model_dense_save_5/dense_9255/MatMul/ReadVariableOp6my_model_dense_save_5/dense_9255/MatMul/ReadVariableOp2r
7my_model_dense_save_5/dense_9256/BiasAdd/ReadVariableOp7my_model_dense_save_5/dense_9256/BiasAdd/ReadVariableOp2p
6my_model_dense_save_5/dense_9256/MatMul/ReadVariableOp6my_model_dense_save_5/dense_9256/MatMul/ReadVariableOp:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�	
�
J__inference_dense_9255_layer_call_and_return_conditional_losses_1217095235

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_1217095066	
state
dense_9254_1217095049
dense_9254_1217095051
dense_9255_1217095054
dense_9255_1217095056
dense_9256_1217095059
dense_9256_1217095061

identity_1

identity_2��"dense_9254/StatefulPartitionedCall�"dense_9255/StatefulPartitionedCall�"dense_9256/StatefulPartitionedCall^
IdentityIdentitystate*
T0*,
_output_shapes
:����������2

Identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSlicestatestrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
"dense_9254/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0dense_9254_1217095049dense_9254_1217095051*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_9254_layer_call_and_return_conditional_losses_12170949662$
"dense_9254/StatefulPartitionedCall�
"dense_9255/StatefulPartitionedCallStatefulPartitionedCall+dense_9254/StatefulPartitionedCall:output:0dense_9255_1217095054dense_9255_1217095056*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_9255_layer_call_and_return_conditional_losses_12170949932$
"dense_9255/StatefulPartitionedCall�
"dense_9256/StatefulPartitionedCallStatefulPartitionedCall+dense_9255/StatefulPartitionedCall:output:0dense_9256_1217095059dense_9256_1217095061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_9256_layer_call_and_return_conditional_losses_12170950202$
"dense_9256/StatefulPartitionedCall�

Identity_1Identity+dense_9256/StatefulPartitionedCall:output:0#^dense_9254/StatefulPartitionedCall#^dense_9255/StatefulPartitionedCall#^dense_9256/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity+dense_9256/StatefulPartitionedCall:output:0#^dense_9254/StatefulPartitionedCall#^dense_9255/StatefulPartitionedCall#^dense_9256/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*C
_input_shapes2
0:����������::::::2H
"dense_9254/StatefulPartitionedCall"dense_9254/StatefulPartitionedCall2H
"dense_9255/StatefulPartitionedCall"dense_9255/StatefulPartitionedCall2H
"dense_9256/StatefulPartitionedCall"dense_9256/StatefulPartitionedCall:S O
,
_output_shapes
:����������

_user_specified_namestate
�
�
/__inference_dense_9255_layer_call_fn_1217095244

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_dense_9255_layer_call_and_return_conditional_losses_12170949932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference__traced_restore_1217095334
file_prefix<
8assignvariableop_my_model_dense_save_5_dense_9254_kernel<
8assignvariableop_1_my_model_dense_save_5_dense_9254_bias>
:assignvariableop_2_my_model_dense_save_5_dense_9255_kernel<
8assignvariableop_3_my_model_dense_save_5_dense_9255_bias>
:assignvariableop_4_my_model_dense_save_5_dense_9256_kernel<
8assignvariableop_5_my_model_dense_save_5_dense_9256_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp8assignvariableop_my_model_dense_save_5_dense_9254_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp8assignvariableop_1_my_model_dense_save_5_dense_9254_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp:assignvariableop_2_my_model_dense_save_5_dense_9255_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp8assignvariableop_3_my_model_dense_save_5_dense_9255_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp:assignvariableop_4_my_model_dense_save_5_dense_9256_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp8assignvariableop_5_my_model_dense_save_5_dense_9256_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
J__inference_dense_9256_layer_call_and_return_conditional_losses_1217095020

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
J__inference_dense_9255_layer_call_and_return_conditional_losses_1217094993

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
@
input_15
serving_default_input_1:0����������<
output_10
StatefulPartitionedCall:0���������<
output_20
StatefulPartitionedCall:1���������tensorflow/serving/predict:�\
�
	my_layers
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
*2&call_and_return_all_conditional_losses
3_default_save_signature
4__call__"�
_tf_keras_model�{"class_name": "MyModel_dense_save", "name": "my_model_dense_save_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MyModel_dense_save"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
5
	0

1
2"
trackable_list_wrapper
"
	optimizer
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
layer_regularization_losses
	variables
non_trainable_variables
layer_metrics
trainable_variables

layers
metrics
regularization_losses
4__call__
3_default_save_signature
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
,
5serving_default"
signature_map
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*6&call_and_return_all_conditional_losses
7__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_9254", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9254", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 250}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 250]}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*8&call_and_return_all_conditional_losses
9__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_9255", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9255", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 200]}}
�

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
*:&call_and_return_all_conditional_losses
;__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_9256", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9256", "trainable": true, "dtype": "float32", "units": 3, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 200]}}
;:9
��2'my_model_dense_save_5/dense_9254/kernel
4:2�2%my_model_dense_save_5/dense_9254/bias
;:9
��2'my_model_dense_save_5/dense_9255/kernel
4:2�2%my_model_dense_save_5/dense_9255/bias
::8	�2'my_model_dense_save_5/dense_9256/kernel
3:12%my_model_dense_save_5/dense_9256/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
	0

1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
#non_trainable_variables
$layer_metrics
trainable_variables

%layers
regularization_losses
&metrics
'layer_regularization_losses
7__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
(non_trainable_variables
)layer_metrics
trainable_variables

*layers
regularization_losses
+metrics
,layer_regularization_losses
9__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
-non_trainable_variables
.layer_metrics
 trainable_variables

/layers
!regularization_losses
0metrics
1layer_regularization_losses
;__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_1217095185
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_1217095135�
���
FullArgSpec.
args&�#
jself
jstate
jinitial_states
varargs
 
varkw
 $
defaults�
�

 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference__wrapped_model_1217094946�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *+�(
&�#
input_1����������
�2�
:__inference_my_model_dense_save_5_layer_call_fn_1217095154
:__inference_my_model_dense_save_5_layer_call_fn_1217095204�
���
FullArgSpec.
args&�#
jself
jstate
jinitial_states
varargs
 
varkw
 $
defaults�
�

 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_signature_wrapper_1217095104input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_9254_layer_call_and_return_conditional_losses_1217095215�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_9254_layer_call_fn_1217095224�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_9255_layer_call_and_return_conditional_losses_1217095235�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_9255_layer_call_fn_1217095244�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_9256_layer_call_and_return_conditional_losses_1217095255�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_9256_layer_call_fn_1217095264�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
%__inference__wrapped_model_1217094946�5�2
+�(
&�#
input_1����������
� "c�`
.
output_1"�
output_1���������
.
output_2"�
output_2����������
J__inference_dense_9254_layer_call_and_return_conditional_losses_1217095215^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
/__inference_dense_9254_layer_call_fn_1217095224Q0�-
&�#
!�
inputs����������
� "������������
J__inference_dense_9255_layer_call_and_return_conditional_losses_1217095235^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
/__inference_dense_9255_layer_call_fn_1217095244Q0�-
&�#
!�
inputs����������
� "������������
J__inference_dense_9256_layer_call_and_return_conditional_losses_1217095255]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
/__inference_dense_9256_layer_call_fn_1217095264P0�-
&�#
!�
inputs����������
� "�����������
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_1217095135�H�E
>�;
$�!
state����������
�

 

 

 

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
U__inference_my_model_dense_save_5_layer_call_and_return_conditional_losses_1217095185�J�G
@�=
&�#
input_1����������
�

 

 

 

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
:__inference_my_model_dense_save_5_layer_call_fn_1217095154�H�E
>�;
$�!
state����������
�

 

 

 

 
� "=�:
�
0���������
�
1����������
:__inference_my_model_dense_save_5_layer_call_fn_1217095204�J�G
@�=
&�#
input_1����������
�

 

 

 

 
� "=�:
�
0���������
�
1����������
(__inference_signature_wrapper_1217095104�@�=
� 
6�3
1
input_1&�#
input_1����������"c�`
.
output_1"�
output_1���������
.
output_2"�
output_2���������