ķĪ 
ó"Č"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint’’’’’’’’’"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Į
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
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018Ń
®
+adversary_env/ValueRnnNetwork/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+adversary_env/ValueRnnNetwork/dense_17/bias
§
?adversary_env/ValueRnnNetwork/dense_17/bias/Read/ReadVariableOpReadVariableOp+adversary_env/ValueRnnNetwork/dense_17/bias*
_output_shapes
:*
dtype0
¶
-adversary_env/ValueRnnNetwork/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*>
shared_name/-adversary_env/ValueRnnNetwork/dense_17/kernel
Æ
Aadversary_env/ValueRnnNetwork/dense_17/kernel/Read/ReadVariableOpReadVariableOp-adversary_env/ValueRnnNetwork/dense_17/kernel*
_output_shapes

:(*
dtype0
ß
Cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *T
shared_nameECadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias
Ų
Wadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias/Read/ReadVariableOpReadVariableOpCadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias*
_output_shapes	
: *
dtype0
ū
Oadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	( *`
shared_nameQOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel
ō
cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel/Read/ReadVariableOpReadVariableOpOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel*
_output_shapes
:	( *
dtype0
ē
Eadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *V
shared_nameGEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel
ą
Yadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel/Read/ReadVariableOpReadVariableOpEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel*
_output_shapes
:	  *
dtype0
ī
Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *\
shared_nameMKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias
ē
_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias/Read/ReadVariableOpReadVariableOpKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias*
_output_shapes
: *
dtype0
ö
Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *^
shared_nameOMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel
ļ
aadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel/Read/ReadVariableOpReadVariableOpMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel*
_output_shapes

:  *
dtype0
ī
Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *\
shared_nameMKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias
ē
_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias/Read/ReadVariableOpReadVariableOpKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias*
_output_shapes
: *
dtype0
÷
Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ģ *^
shared_nameOMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel
š
aadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel/Read/ReadVariableOpReadVariableOpMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel*
_output_shapes
:	Ģ *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:
*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:4
* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:4
*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
ż
Radversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:©*c
shared_nameTRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
ö
fadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpReadVariableOpRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
_output_shapes	
:©*
dtype0

Tadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
©*e
shared_nameVTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
’
hadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpReadVariableOpTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel* 
_output_shapes
:
©*
dtype0

[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*l
shared_name][adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias

oadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias/Read/ReadVariableOpReadVariableOp[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias*
_output_shapes	
:*
dtype0
¬
gadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*x
shared_nameigadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel
„
{adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel/Read/ReadVariableOpReadVariableOpgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel* 
_output_shapes
:
*
dtype0

]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *n
shared_name_]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel

qadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel/Read/ReadVariableOpReadVariableOp]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel*
_output_shapes
:	 *
dtype0

cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *t
shared_nameecadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias

wadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias/Read/ReadVariableOpReadVariableOpcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias*
_output_shapes
: *
dtype0
¦
eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *v
shared_namegeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel

yadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel/Read/ReadVariableOpReadVariableOpeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel*
_output_shapes

:  *
dtype0

cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *t
shared_nameecadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias

wadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOpReadVariableOpcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias*
_output_shapes
: *
dtype0
§
eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ģ *v
shared_namegeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel
 
yadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOpReadVariableOpeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel*
_output_shapes
:	Ģ *
dtype0
v
dense_12/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_12/bias_1
o
#dense_12/bias_1/Read/ReadVariableOpReadVariableOpdense_12/bias_1*
_output_shapes
:
*
dtype0
~
dense_12/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:4
*"
shared_namedense_12/kernel_1
w
%dense_12/kernel_1/Read/ReadVariableOpReadVariableOpdense_12/kernel_1*
_output_shapes

:4
*
dtype0
v
conv2d_2/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/bias_1
o
#conv2d_2/bias_1/Read/ReadVariableOpReadVariableOpconv2d_2/bias_1*
_output_shapes
:*
dtype0

conv2d_2/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_2/kernel_1

%conv2d_2/kernel_1/Read/ReadVariableOpReadVariableOpconv2d_2/kernel_1*&
_output_shapes
:*
dtype0
j
global_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	

NoOpNoOp
»ē
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*õę
valueźęBęę BŽę
į
collect_data_spec
policy_state_spec

train_step
metadata
model_variables
_all_assets

action
distribution
	get_initial_state

get_metadata
get_train_step

signatures*

observation
1* 

actor_network_state* 
JD
VARIABLE_VALUEglobal_step%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ź
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
'24
(25*
s
)_time_step_spec
*_policy_state_spec
+_policy_step_spec
,_trajectory_spec
-_wrapped_policy*

.trace_0
/trace_1* 

0trace_0* 

1trace_0* 
* 
* 
K

2action
3get_initial_state
4get_train_step
5get_metadata* 
* 
* 
WQ
VARIABLE_VALUEconv2d_2/kernel_1,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv2d_2/bias_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdense_12/kernel_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_12/bias_1,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
¬„
VARIABLE_VALUEeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
Ŗ£
VARIABLE_VALUEcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
¬„
VARIABLE_VALUEeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
Ŗ£
VARIABLE_VALUEcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUE]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
®§
VARIABLE_VALUEgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
£
VARIABLE_VALUE[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUETadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_2/kernel-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv2d_2/bias-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdense_12/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_12/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUECadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias-model_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE-adversary_env/ValueRnnNetwork/dense_17/kernel-model_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE+adversary_env/ValueRnnNetwork/dense_17/bias-model_variables/25/.ATTRIBUTES/VARIABLE_VALUE*

observation
3* 

actor_network_state* 

	*state
*1* 

observation
1* 

6_actor_network
7_time_step_spec
8_policy_state_spec
9_policy_step_spec
:_trajectory_spec
;_value_network*
* 
* 
* 
* 
* 
* 
* 
* 
ę
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_input_tensor_spec
C_state_spec
D_lstm_encoder
E_projection_networks*

Fobservation
F3* 

Gactor_network_state* 

	8state
81* 

Fobservation
F1* 
č
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_input_tensor_spec
O_state_spec
P_lstm_encoder
Q_postprocessing_layers*
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
* 

Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 
* 
* 
õ
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_input_tensor_spec
^_state_spec
__input_encoder
`_lstm_network
a_output_encoder*
§
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_projection_layer*
* 
* 
b
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12*
b
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12*
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
* 
* 
* 
* 
õ
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_input_tensor_spec
u_state_spec
v_input_encoder
w_lstm_network
x_output_encoder*
¦
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

'kernel
(bias*
* 

D0
E1*
* 
* 
* 
R
0
1
2
3
4
5
6
7
8
9
10*
R
0
1
2
3
4
5
6
7
8
9
10*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
* 
* 
„
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_input_tensor_spec
_preprocessing_nest
_flat_preprocessing_layers
_preprocessing_combiner
_postprocessing_layers*
”
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	cell*
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

kernel
bias*
* 

P0
Q1*
* 
* 
* 
R
0
1
2
3
 4
!5
"6
#7
$8
%9
&10*
R
0
1
2
3
 4
!5
"6
#7
$8
%9
&10*
* 

”non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
„layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*
* 
* 
* 
* 
„
¦	variables
§trainable_variables
Øregularization_losses
©	keras_api
Ŗ__call__
+«&call_and_return_all_conditional_losses
¬_input_tensor_spec
­_preprocessing_nest
®_flat_preprocessing_layers
Æ_preprocessing_combiner
°_postprocessing_layers*
”
±	variables
²trainable_variables
³regularization_losses
“	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses
	·cell*
* 

'0
(1*

'0
(1*
* 

ønon_trainable_variables
¹layers
ŗmetrics
 »layer_regularization_losses
¼layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
* 
* 
* 

_0
`1*
* 
* 
* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 

½non_trainable_variables
¾layers
æmetrics
 Ąlayer_regularization_losses
Įlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 

Ā0
Ć1
Ä2*

Å	variables
Ętrainable_variables
Ēregularization_losses
Č	keras_api
É__call__
+Ź&call_and_return_all_conditional_losses* 

Ė0
Ģ1
Ķ2*

0
1
2*

0
1
2*
* 

Īnon_trainable_variables
Ļlayers
Šmetrics
 Ńlayer_regularization_losses
Ņlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
ė
Ó	variables
Ōtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ų&call_and_return_all_conditional_losses
Ł_random_generator
Ś
state_size

kernel
recurrent_kernel
bias*
* 

h0*
* 
* 
* 

0
1*

0
1*
* 

Ūnon_trainable_variables
Ülayers
Żmetrics
 Žlayer_regularization_losses
ßlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 

v0
w1*
* 
* 
* 
<
0
1
2
3
 4
!5
"6
#7*
<
0
1
2
3
 4
!5
"6
#7*
* 

ąnon_trainable_variables
įlayers
āmetrics
 ćlayer_regularization_losses
älayer_metrics
¦	variables
§trainable_variables
Øregularization_losses
Ŗ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*
* 
* 
* 
* 

å0
ę1
ē2*

č	variables
étrainable_variables
źregularization_losses
ė	keras_api
ģ__call__
+ķ&call_and_return_all_conditional_losses* 

ī0
ļ1
š2*

$0
%1
&2*

$0
%1
&2*
* 

ńnon_trainable_variables
ņlayers
ómetrics
 ōlayer_regularization_losses
õlayer_metrics
±	variables
²trainable_variables
³regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses*
* 
* 
ė
ö	variables
÷trainable_variables
ųregularization_losses
ł	keras_api
ś__call__
+ū&call_and_return_all_conditional_losses
ü_random_generator
ż
state_size

$kernel
%recurrent_kernel
&bias*
* 
* 
* 
* 
* 
* 
<
Ā0
Ć1
Ä2
3
Ė4
Ģ5
Ķ6*
* 
* 
* 
é
žlayer-0
’layer_with_weights-0
’layer-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¢
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ķ
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Å	variables
Ętrainable_variables
Ēregularization_losses
É__call__
+Ź&call_and_return_all_conditional_losses
'Ź"call_and_return_conditional_losses* 
* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+”&call_and_return_all_conditional_losses* 
¬
¢	variables
£trainable_variables
¤regularization_losses
„	keras_api
¦__call__
+§&call_and_return_all_conditional_losses

kernel
bias*
¬
Ø	variables
©trainable_variables
Ŗregularization_losses
«	keras_api
¬__call__
+­&call_and_return_all_conditional_losses

kernel
bias*
* 

0*
* 
* 
* 

0
1
2*

0
1
2*
* 

®non_trainable_variables
Ælayers
°metrics
 ±layer_regularization_losses
²layer_metrics
Ó	variables
Ōtrainable_variables
Õregularization_losses
×__call__
+Ų&call_and_return_all_conditional_losses
'Ų"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
å0
ę1
ē2
Æ3
ī4
ļ5
š6*
* 
* 
* 
é
³layer-0
“layer_with_weights-0
“layer-1
µlayer-2
¶layer-3
·	variables
øtrainable_variables
¹regularization_losses
ŗ	keras_api
»__call__
+¼&call_and_return_all_conditional_losses*
¢
½layer-0
¾	variables
ætrainable_variables
Ąregularization_losses
Į	keras_api
Ā__call__
+Ć&call_and_return_all_conditional_losses* 
Ķ
Älayer-0
Ålayer_with_weights-0
Ålayer-1
Ę	variables
Ētrainable_variables
Čregularization_losses
É	keras_api
Ź__call__
+Ė&call_and_return_all_conditional_losses*
* 
* 
* 

Ģnon_trainable_variables
Ķlayers
Īmetrics
 Ļlayer_regularization_losses
Šlayer_metrics
č	variables
étrainable_variables
źregularization_losses
ģ__call__
+ķ&call_and_return_all_conditional_losses
'ķ"call_and_return_conditional_losses* 
* 
* 

Ń	variables
Ņtrainable_variables
Óregularization_losses
Ō	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses* 
¬
×	variables
Ųtrainable_variables
Łregularization_losses
Ś	keras_api
Ū__call__
+Ü&call_and_return_all_conditional_losses

 kernel
!bias*
¬
Ż	variables
Žtrainable_variables
ßregularization_losses
ą	keras_api
į__call__
+ā&call_and_return_all_conditional_losses

"kernel
#bias*
* 

·0*
* 
* 
* 

$0
%1
&2*

$0
%1
&2*
* 

ćnon_trainable_variables
älayers
åmetrics
 ęlayer_regularization_losses
ēlayer_metrics
ö	variables
÷trainable_variables
ųregularization_losses
ś__call__
+ū&call_and_return_all_conditional_losses
'ū"call_and_return_conditional_losses*
* 
* 
* 
* 

č	variables
étrainable_variables
źregularization_losses
ė	keras_api
ģ__call__
+ķ&call_and_return_all_conditional_losses* 
Ļ
ī	variables
ļtrainable_variables
šregularization_losses
ń	keras_api
ņ__call__
+ó&call_and_return_all_conditional_losses

kernel
bias
!ō_jit_compiled_convolution_op*

õ	variables
ötrainable_variables
÷regularization_losses
ų	keras_api
ł__call__
+ś&call_and_return_all_conditional_losses* 

ū	variables
ütrainable_variables
żregularization_losses
ž	keras_api
’__call__
+&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
 trace_3* 

”	variables
¢trainable_variables
£regularization_losses
¤	keras_api
„__call__
+¦&call_and_return_all_conditional_losses* 
¬
§	variables
Øtrainable_variables
©regularization_losses
Ŗ	keras_api
«__call__
+¬&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 

­non_trainable_variables
®layers
Æmetrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
:
²trace_0
³trace_1
“trace_2
µtrace_3* 
:
¶trace_0
·trace_1
øtrace_2
¹trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 

ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 

ænon_trainable_variables
Ąlayers
Įmetrics
 Ālayer_regularization_losses
Ćlayer_metrics
¢	variables
£trainable_variables
¤regularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

Änon_trainable_variables
Ålayers
Ęmetrics
 Ēlayer_regularization_losses
Člayer_metrics
Ø	variables
©trainable_variables
Ŗregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

É	variables
Źtrainable_variables
Ėregularization_losses
Ģ	keras_api
Ķ__call__
+Ī&call_and_return_all_conditional_losses* 
Ļ
Ļ	variables
Štrainable_variables
Ńregularization_losses
Ņ	keras_api
Ó__call__
+Ō&call_and_return_all_conditional_losses

kernel
bias
!Õ_jit_compiled_convolution_op*

Ö	variables
×trainable_variables
Ųregularization_losses
Ł	keras_api
Ś__call__
+Ū&call_and_return_all_conditional_losses* 

Ü	variables
Żtrainable_variables
Žregularization_losses
ß	keras_api
ą__call__
+į&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 

ānon_trainable_variables
ćlayers
ämetrics
 ålayer_regularization_losses
ęlayer_metrics
·	variables
øtrainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses*
:
ētrace_0
čtrace_1
étrace_2
źtrace_3* 
:
ėtrace_0
ģtrace_1
ķtrace_2
ītrace_3* 

ļ	variables
štrainable_variables
ńregularization_losses
ņ	keras_api
ó__call__
+ō&call_and_return_all_conditional_losses* 
* 
* 
* 

õnon_trainable_variables
ölayers
÷metrics
 ųlayer_regularization_losses
łlayer_metrics
¾	variables
ætrainable_variables
Ąregularization_losses
Ā__call__
+Ć&call_and_return_all_conditional_losses
'Ć"call_and_return_conditional_losses* 
:
śtrace_0
ūtrace_1
ütrace_2
żtrace_3* 
:
žtrace_0
’trace_1
trace_2
trace_3* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ę	variables
Ētrainable_variables
Čregularization_losses
Ź__call__
+Ė&call_and_return_all_conditional_losses
'Ė"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ń	variables
Ņtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses* 
* 
* 

 0
!1*

 0
!1*
* 

 non_trainable_variables
”layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
×	variables
Ųtrainable_variables
Łregularization_losses
Ū__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses*
* 
* 

"0
#1*

"0
#1*
* 

„non_trainable_variables
¦layers
§metrics
 Ølayer_regularization_losses
©layer_metrics
Ż	variables
Žtrainable_variables
ßregularization_losses
į__call__
+ā&call_and_return_all_conditional_losses
'ā"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ŗnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
č	variables
étrainable_variables
źregularization_losses
ģ__call__
+ķ&call_and_return_all_conditional_losses
'ķ"call_and_return_conditional_losses* 

Ætrace_0
°trace_1* 

±trace_0
²trace_1* 

0
1*

0
1*
* 

³non_trainable_variables
“layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
ī	variables
ļtrainable_variables
šregularization_losses
ņ__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses*

øtrace_0* 

¹trace_0* 
* 
* 
* 
* 

ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ł__call__
+ś&call_and_return_all_conditional_losses
'ś"call_and_return_conditional_losses* 

ætrace_0* 

Ątrace_0* 
* 
* 
* 

Įnon_trainable_variables
Ālayers
Ćmetrics
 Älayer_regularization_losses
Ålayer_metrics
ū	variables
ütrainable_variables
żregularization_losses
’__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Ętrace_0* 

Ētrace_0* 
* 
$
ž0
’1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Čnon_trainable_variables
Élayers
Źmetrics
 Ėlayer_regularization_losses
Ģlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Ķtrace_0
Ītrace_1* 

Ļtrace_0
Štrace_1* 
* 


0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ńnon_trainable_variables
Ņlayers
Ómetrics
 Ōlayer_regularization_losses
Õlayer_metrics
”	variables
¢trainable_variables
£regularization_losses
„__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 

Ötrace_0
×trace_1* 

Ųtrace_0
Łtrace_1* 

0
1*

0
1*
* 

Śnon_trainable_variables
Ūlayers
Ümetrics
 Żlayer_regularization_losses
Žlayer_metrics
§	variables
Øtrainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*

ßtrace_0* 

ątrace_0* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

įnon_trainable_variables
ālayers
ćmetrics
 älayer_regularization_losses
ålayer_metrics
É	variables
Źtrainable_variables
Ėregularization_losses
Ķ__call__
+Ī&call_and_return_all_conditional_losses
'Ī"call_and_return_conditional_losses* 

ętrace_0
ētrace_1* 

čtrace_0
étrace_1* 

0
1*

0
1*
* 

źnon_trainable_variables
ėlayers
ģmetrics
 ķlayer_regularization_losses
īlayer_metrics
Ļ	variables
Štrainable_variables
Ńregularization_losses
Ó__call__
+Ō&call_and_return_all_conditional_losses
'Ō"call_and_return_conditional_losses*

ļtrace_0* 

štrace_0* 
* 
* 
* 
* 

ńnon_trainable_variables
ņlayers
ómetrics
 ōlayer_regularization_losses
õlayer_metrics
Ö	variables
×trainable_variables
Ųregularization_losses
Ś__call__
+Ū&call_and_return_all_conditional_losses
'Ū"call_and_return_conditional_losses* 

ötrace_0* 

÷trace_0* 
* 
* 
* 

ųnon_trainable_variables
łlayers
śmetrics
 ūlayer_regularization_losses
ülayer_metrics
Ü	variables
Żtrainable_variables
Žregularization_losses
ą__call__
+į&call_and_return_all_conditional_losses
'į"call_and_return_conditional_losses* 

żtrace_0* 

žtrace_0* 
* 
$
³0
“1
µ2
¶3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

’non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ļ	variables
štrainable_variables
ńregularization_losses
ó__call__
+ō&call_and_return_all_conditional_losses
'ō"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 


½0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

Ä0
Å1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
l
action_0_discountPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

action_0_observation_imagePlaceholder*/
_output_shapes
:’’’’’’’’’*
dtype0*$
shape:’’’’’’’’’

action_0_observation_random_zPlaceholder*'
_output_shapes
:’’’’’’’’’2*
dtype0*
shape:’’’’’’’’’2

action_0_observation_time_stepPlaceholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
j
action_0_rewardPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
m
action_0_step_typePlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

action_1_actor_network_state_0Placeholder*(
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

action_1_actor_network_state_1Placeholder*(
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
ć
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_imageaction_0_observation_random_zaction_0_observation_time_stepaction_0_rewardaction_0_step_typeaction_1_actor_network_state_0action_1_actor_network_state_1conv2d_2/kernel_1conv2d_2/bias_1dense_12/kernel_1dense_12/bias_1eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/biaseadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernelgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/biasTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias* 
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_9448758
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¼
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_9448767
Ū
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_9448779

StatefulPartitionedCall_1StatefulPartitionedCallglobal_step*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_9448775
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ą
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameglobal_step/Read/ReadVariableOp%conv2d_2/kernel_1/Read/ReadVariableOp#conv2d_2/bias_1/Read/ReadVariableOp%dense_12/kernel_1/Read/ReadVariableOp#dense_12/bias_1/Read/ReadVariableOpyadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOpwadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOpyadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel/Read/ReadVariableOpwadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias/Read/ReadVariableOpqadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel/Read/ReadVariableOp{adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel/Read/ReadVariableOpoadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias/Read/ReadVariableOphadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpfadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpaadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel/Read/ReadVariableOp_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias/Read/ReadVariableOpaadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel/Read/ReadVariableOp_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias/Read/ReadVariableOpYadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel/Read/ReadVariableOpcadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel/Read/ReadVariableOpWadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias/Read/ReadVariableOpAadversary_env/ValueRnnNetwork/dense_17/kernel/Read/ReadVariableOp?adversary_env/ValueRnnNetwork/dense_17/bias/Read/ReadVariableOpConst*(
Tin!
2	*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_9450096

StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameglobal_stepconv2d_2/kernel_1conv2d_2/bias_1dense_12/kernel_1dense_12/bias_1eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/biaseadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernelgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/biasTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasconv2d_2/kernelconv2d_2/biasdense_12/kerneldense_12/biasMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernelKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/biasMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernelKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/biasEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernelOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernelCadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias-adversary_env/ValueRnnNetwork/dense_17/kernel+adversary_env/ValueRnnNetwork/dense_17/bias*'
Tin 
2*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_9450187ĀÓ
Č
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449175

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Š
e
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448981

inputs
identity¹
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9448966i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
č
m
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448999
lambda_6_input
identityĮ
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9448966i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:W S
'
_output_shapes
:’’’’’’’’’2
(
_user_specified_namelambda_6_input
Ņ
-
+__inference_function_with_signature_1949163ć
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_<lambda>_3492*(
_construction_contextkEagerRuntime*
_input_shapes 
Ø
×
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449120
lambda_5_input"
dense_12_9449114:4

dense_12_9449116:

identity¢ dense_12/StatefulPartitionedCallĮ
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449016
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_9449114dense_12_9449116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449028x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_5_input
Z

__inference_<lambda>_3492*(
_construction_contextkEagerRuntime*
_input_shapes 
ł
e
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449667

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
Ø
×
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449130
lambda_5_input"
dense_12_9449124:4

dense_12_9449126:

identity¢ dense_12/StatefulPartitionedCallĮ
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449070
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_9449124dense_12_9449126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449028x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_5_input

ß
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449276
lambda_4_input*
conv2d_2_9449268:
conv2d_2_9449270:
identity¢ conv2d_2/StatefulPartitionedCallÉ
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449144
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_9449268conv2d_2_9449270*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449156ā
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449167Ö
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449175r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_4_input

F
*__inference_lambda_5_layer_call_fn_9449940

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449421`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ś
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449016

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ø
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦K
§
 __inference__traced_save_9450096
file_prefix*
&savev2_global_step_read_readvariableop	0
,savev2_conv2d_2_kernel_1_read_readvariableop.
*savev2_conv2d_2_bias_1_read_readvariableop0
,savev2_dense_12_kernel_1_read_readvariableop.
*savev2_dense_12_bias_1_read_readvariableop
savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel_read_readvariableop
~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias_read_readvariableop
savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel_read_readvariableop
~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias_read_readvariableop|
xsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel_read_readvariableop
savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel_read_readvariableopz
vsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_bias_read_readvariableops
osavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopq
msavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableopl
hsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernel_read_readvariableopj
fsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_bias_read_readvariableopl
hsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernel_read_readvariableopj
fsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_bias_read_readvariableopd
`savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernel_read_readvariableopn
jsavev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernel_read_readvariableopb
^savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_bias_read_readvariableopL
Hsavev2_adversary_env_valuernnnetwork_dense_17_kernel_read_readvariableopJ
Fsavev2_adversary_env_valuernnnetwork_dense_17_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ķ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

value
B
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH„
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_global_step_read_readvariableop,savev2_conv2d_2_kernel_1_read_readvariableop*savev2_conv2d_2_bias_1_read_readvariableop,savev2_dense_12_kernel_1_read_readvariableop*savev2_dense_12_bias_1_read_readvariableopsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel_read_readvariableop~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias_read_readvariableopsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel_read_readvariableop~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias_read_readvariableopxsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel_read_readvariableopsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel_read_readvariableopvsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_bias_read_readvariableoposavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopmsavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableophsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernel_read_readvariableopfsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_bias_read_readvariableophsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernel_read_readvariableopfsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_bias_read_readvariableop`savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernel_read_readvariableopjsavev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernel_read_readvariableop^savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_bias_read_readvariableopHsavev2_adversary_env_valuernnnetwork_dense_17_kernel_read_readvariableopFsavev2_adversary_env_valuernnnetwork_dense_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesł
ö: : :::4
:
:	Ģ : :  : :	 :
::
©:©:::4
:
:	Ģ : :  : :	  :	( : :(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:4
: 

_output_shapes
:
:%!

_output_shapes
:	Ģ : 

_output_shapes
: :$ 

_output_shapes

:  : 	

_output_shapes
: :%
!

_output_shapes
:	 :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
©:!

_output_shapes	
:©:,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:4
: 

_output_shapes
:
:%!

_output_shapes
:	Ģ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :%!

_output_shapes
:	  :%!

_output_shapes
:	( :!

_output_shapes	
: :$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: 

F
*__inference_lambda_6_layer_call_fn_9449786

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9448948`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
æ
F
*__inference_lambda_4_layer_call_fn_9449853

inputs
identityø
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449144h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
õ
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449795

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
č
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449770

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
§
J
.__inference_sequential_6_layer_call_fn_9449536

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448951`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ž¢
Į
__inference_action_1949408
	step_type

reward
discount
observation_image
observation_random_z
observation_time_step
actor_network_state_0
actor_network_state_1
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4

}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:

oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	Ģ ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
©n
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	©
identity	

identity_1

identity_2¢gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp¢fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp¢gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp¢fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp¢tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp¢sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp¢tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp¢sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp¢kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp¢jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp¢lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp¢VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp¢UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
ShapeShapediscount*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’I
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’F
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: @
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’b
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:b
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:’’’’’’’’’
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0*
T0*(
_output_shapes
:’’’’’’’’’

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_1*
T0*(
_output_shapes
:’’’’’’’’’?
Shape_2Shapediscount*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:\
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’K
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :O
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: B
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:g
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’h
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:N
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :c
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : }
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:f
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:’’’’’’’’’

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ņ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:’’’’’’’’’
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :ń
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ņ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimsobservation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :ā
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:’’’’’’’’’ę
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	¼
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’         Ż
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	¶
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’2   Ū
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	¶
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ū
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’¬
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’ø
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¹
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
®
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’µ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’³
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?“
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4¼
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4°
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
®
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
£
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ū
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’ĢØ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’Ģ
  ą
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ģ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	Ģ *
dtype0ē
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ ņ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0č
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ ņ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ µ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¹
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ¹
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask„
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : č
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ü
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:’’’’’’’’’ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:©
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¤
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:Ł
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’ Ō
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:¦
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ø
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ø
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¾
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskŖ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ¹
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:’’’’’’’’’
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ī
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Č
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ņ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ī
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’ü
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 ü
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 ×
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ū
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ē
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¤
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ä
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:’’’’’’’’’
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ķ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¦
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¹
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split’
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’Ņ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’ł
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’Õ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’Ō
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’ö
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ł
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ś
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:’’’’’’’’’ņ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
squeeze_dims
ö
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
©*
dtype0¬
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’©ó
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:©*
dtype0·
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’©
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’©   ”
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’©“
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ö
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:’’’’’’’’’T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ²
Deterministic/sample/ShapeShapehCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ń
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:’’’’’’’’’u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ø
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:’’’’’’’’’Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 RØ
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:’’’’’’’’’Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:’’’’’’’’’\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:’’’’’’’’’°

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’°

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’Į
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Å
_input_shapes³
°:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’2:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : 2Ņ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2Š
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2Ņ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2Š
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2ģ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2ź
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2ģ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2ź
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2Ś
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2Ų
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2Ü
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2°
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2®
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	step_type:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_namereward:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
discount:b^
/
_output_shapes
:’’’’’’’’’
+
_user_specified_nameobservation/image:]Y
'
_output_shapes
:’’’’’’’’’2
.
_user_specified_nameobservation/random_z:^Z
'
_output_shapes
:’’’’’’’’’
/
_user_specified_nameobservation/time_step:_[
(
_output_shapes
:’’’’’’’’’
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:’’’’’’’’’
/
_user_specified_nameactor_network_state/1
Č
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449781

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
æ
F
*__inference_lambda_4_layer_call_fn_9449722

inputs
identityø
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9448793h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
č
m
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449345
lambda_6_input
identityĮ
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449299i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:W S
'
_output_shapes
:’’’’’’’’’2
(
_user_specified_namelambda_6_input
æ
F
*__inference_lambda_4_layer_call_fn_9449727

inputs
identityø
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9448871h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

¦
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449531

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’W
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ŗ
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’i
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’j
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
³
G
+__inference_flatten_6_layer_call_fn_9449775

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9448824a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Č	
ö
E__inference_dense_12_layer_call_and_return_conditional_losses_9449979

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’4
 
_user_specified_nameinputs
§
J
.__inference_sequential_6_layer_call_fn_9449654

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449302`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
č
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9448816

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ę
£
.__inference_sequential_4_layer_call_fn_9449608

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449178p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Š
e
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448951

inputs
identity¹
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9448948i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ä
£
.__inference_sequential_5_layer_call_fn_9449461
lambda_5_input
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449445o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_5_input
§
J
.__inference_sequential_6_layer_call_fn_9449659

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449332`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
Š
e
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449302

inputs
identity¹
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449299i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
Ä

*__inference_dense_12_layer_call_fn_9449838

inputs
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449028o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’4: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’4
 
_user_specified_nameinputs
¢
 
__inference_action_1949035
	time_step
time_step_1
time_step_2
time_step_3
time_step_4
time_step_5
policy_state
policy_state_1
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4

}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:

oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	Ģ ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
©n
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	©
identity	

identity_1

identity_2¢gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp¢fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp¢gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp¢fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp¢tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp¢sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp¢tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp¢sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp¢kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp¢jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp¢lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp¢VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp¢UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp@
ShapeShapetime_step_2*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’I
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	time_stepEqual/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’F
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: @
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’b
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:b
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:’’’’’’’’’w
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state*
T0*(
_output_shapes
:’’’’’’’’’}

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*(
_output_shapes
:’’’’’’’’’B
Shape_2Shapetime_step_2*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:\
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’K
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	time_stepEqual_1/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :O
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: B
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:g
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’h
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:N
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :c
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : }
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:f
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:’’’’’’’’’

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ģ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_3OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:’’’’’’’’’
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :č
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_4QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :č
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimstime_step_5QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :ā
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:’’’’’’’’’ę
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	¼
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’         Ż
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	¶
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’2   Ū
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	¶
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ū
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’¬
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’ø
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¹
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
®
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’µ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’³
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?“
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4¼
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4°
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
®
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
£
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ū
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’ĢØ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’Ģ
  ą
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ģ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	Ģ *
dtype0ē
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ ņ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0č
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ ņ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ µ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¹
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ¹
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask„
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : č
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ü
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:’’’’’’’’’ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:©
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¤
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:Ł
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’ Ō
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:¦
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ø
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ø
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¾
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskŖ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ¹
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:’’’’’’’’’
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ī
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Č
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ņ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ī
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’ü
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 ü
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 ×
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ū
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ē
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¤
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ä
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:’’’’’’’’’
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ķ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¦
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¹
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split’
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’Ņ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’ł
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’Õ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’Ō
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’ö
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ł
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ś
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:’’’’’’’’’ņ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
squeeze_dims
ö
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
©*
dtype0¬
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’©ó
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:©*
dtype0·
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’©
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’©   ”
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’©“
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ö
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:’’’’’’’’’T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ²
Deterministic/sample/ShapeShapehCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ń
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:’’’’’’’’’u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ø
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:’’’’’’’’’Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 RØ
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:’’’’’’’’’Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:’’’’’’’’’\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:’’’’’’’’’°

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’°

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’Į
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Å
_input_shapes³
°:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’2:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : 2Ņ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2Š
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2Ņ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2Š
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2ģ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2ź
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2ģ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2ź
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2Ś
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2Ų
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2Ü
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2°
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2®
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:NJ
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:NJ
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:ZV
/
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:RN
'
_output_shapes
:’’’’’’’’’2
#
_user_specified_name	time_step:RN
'
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:VR
(
_output_shapes
:’’’’’’’’’
&
_user_specified_namepolicy_state:VR
(
_output_shapes
:’’’’’’’’’
&
_user_specified_namepolicy_state

ß
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448937
lambda_4_input*
conv2d_2_9448929:
conv2d_2_9448931:
identity¢ conv2d_2/StatefulPartitionedCallÉ
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9448871
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_9448929conv2d_2_9448931*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9448805ā
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9448816Ö
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9448824r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_4_input
ś
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449421

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ø
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
÷

I__inference_sequential_5_layer_call_and_return_conditional_losses_9449583

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identity¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ģ
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’

dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’

NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ø
×
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449471
lambda_5_input"
dense_12_9449465:4

dense_12_9449467:

identity¢ dense_12/StatefulPartitionedCallĮ
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449367
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_9449465dense_12_9449467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449379x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_5_input
Ģ

.__inference_sequential_5_layer_call_fn_9449676

inputs
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449386o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ó
`
__inference_<lambda>_3489!
readvariableop_resource:	 
identity	¢ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
©

ž
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9448805

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
õ
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_9448966

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs

F
*__inference_lambda_6_layer_call_fn_9449791

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9448966`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs

F
*__inference_lambda_5_layer_call_fn_9449804

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449016`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

¦
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449633

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’W
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ŗ
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’i
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’j
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
½
E
)__inference_re_lu_2_layer_call_fn_9449765

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9448816h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ä
£
.__inference_sequential_5_layer_call_fn_9449393
lambda_5_input
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449386o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_5_input
ł
e
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449545

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
č
m
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448994
lambda_6_input
identityĮ
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9448948i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:W S
'
_output_shapes
:’’’’’’’’’2
(
_user_specified_namelambda_6_input
æ
F
*__inference_lambda_4_layer_call_fn_9449858

inputs
identityø
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449222h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ß
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449288
lambda_4_input*
conv2d_2_9449280:
conv2d_2_9449282:
identity¢ conv2d_2/StatefulPartitionedCallÉ
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449222
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_9449280conv2d_2_9449282*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449156ā
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449167Ö
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449175r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_4_input
ä
£
.__inference_sequential_5_layer_call_fn_9449110
lambda_5_input
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_5_input

¦
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449649

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’W
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ŗ
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’i
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’j
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Ļ
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449445

inputs"
dense_12_9449439:4

dense_12_9449441:

identity¢ dense_12/StatefulPartitionedCall¹
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449421
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_9449439dense_12_9449441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449379x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
õ
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_9448948

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
­
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449872

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
š
×
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448827

inputs*
conv2d_2_9448806:
conv2d_2_9448808:
identity¢ conv2d_2/StatefulPartitionedCallĮ
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9448793
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_9448806conv2d_2_9448808*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9448805ā
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9448816Ö
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9448824r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

U
%__inference_get_initial_state_1949131

batch_size
identity

identity_1H
packedPack
batch_size*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’W
IdentityIdentityzeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identityzeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
§
J
.__inference_sequential_6_layer_call_fn_9449541

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448981`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs

F
*__inference_lambda_6_layer_call_fn_9449917

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449299`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs

F
*__inference_lambda_6_layer_call_fn_9449922

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449317`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ś
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449367

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ø
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ģ

.__inference_sequential_5_layer_call_fn_9449685

inputs
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449445o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
æ
R
.__inference_sequential_6_layer_call_fn_9448954
lambda_6_input
identity¼
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448951`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:W S
'
_output_shapes
:’’’’’’’’’2
(
_user_specified_namelambda_6_input
õ
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449930

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
õ
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449926

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
­
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449865

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ž
«
.__inference_sequential_4_layer_call_fn_9448913
lambda_4_input!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448897p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_4_input
ž
«
.__inference_sequential_4_layer_call_fn_9448834
lambda_4_input!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448827p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_4_input
Ä

*__inference_dense_12_layer_call_fn_9449969

inputs
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449379o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’4: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’4
 
_user_specified_nameinputs
š
×
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449178

inputs*
conv2d_2_9449157:
conv2d_2_9449159:
identity¢ conv2d_2/StatefulPartitionedCallĮ
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449144
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_9449157conv2d_2_9449159*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449156ā
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449167Ö
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449175r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
­
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449144

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
„

__inference_action_1949649
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_image"
time_step_observation_random_z#
time_step_observation_time_step&
"policy_state_actor_network_state_0&
"policy_state_actor_network_state_1
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4

}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:

oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	Ģ ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
©n
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	©
identity	

identity_1

identity_2¢gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp¢fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp¢gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp¢fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp¢tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp¢sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp¢tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp¢sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp¢kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp¢jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp¢lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp¢VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp¢UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpG
ShapeShapetime_step_discount*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’I
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : c
EqualEqualtime_step_step_typeEqual/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’F
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: @
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’b
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:b
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:’’’’’’’’’
SelectV2SelectV2Reshape:output:0zeros:output:0"policy_state_actor_network_state_0*
T0*(
_output_shapes
:’’’’’’’’’

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0"policy_state_actor_network_state_1*
T0*(
_output_shapes
:’’’’’’’’’I
Shape_2Shapetime_step_discount*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:\
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’K
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : g
Equal_1Equaltime_step_step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :O
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: B
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:g
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’h
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:N
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :c
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : }
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:f
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:’’’’’’’’’

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ü
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_observation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:’’’’’’’’’
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :ū
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_observation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ü
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimstime_step_observation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :ģ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDimstime_step_step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:’’’’’’’’’ę
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	¼
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’         Ż
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	¶
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’2   Ū
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	¶
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ū
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’¬
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’ø
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¹
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
®
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’µ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’³
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?“
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4¼
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4°
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
®
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
£
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ū
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’ĢØ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’Ģ
  ą
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ģ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	Ģ *
dtype0ē
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ ņ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0č
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ ņ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ µ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¹
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ¹
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask„
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : č
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ü
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:’’’’’’’’’ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:©
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¤
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:Ł
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’ Ō
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:¦
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ø
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ø
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¾
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskŖ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ¹
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:’’’’’’’’’
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ī
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Č
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ņ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ī
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’ü
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 ü
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 ×
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ū
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ē
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¤
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ä
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:’’’’’’’’’
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ķ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¦
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¹
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split’
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’Ņ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’ł
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’Õ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’Ō
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’ö
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ł
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ś
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:’’’’’’’’’ņ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
squeeze_dims
ö
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
©*
dtype0¬
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’©ó
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:©*
dtype0·
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’©
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’©   ”
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’©“
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ö
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:’’’’’’’’’T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ²
Deterministic/sample/ShapeShapehCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0*
T0	*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ń
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:’’’’’’’’’u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ø
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:’’’’’’’’’Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 RØ
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:’’’’’’’’’Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:’’’’’’’’’\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:’’’’’’’’’°

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’°

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’Į
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Å
_input_shapes³
°:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’2:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : 2Ņ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2Š
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2Ņ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2Š
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2ģ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2ź
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2ģ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2ź
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2Ś
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2Ų
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2Ü
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2°
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2®
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:X T
#
_output_shapes
:’’’’’’’’’
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:’’’’’’’’’
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:’’’’’’’’’
,
_user_specified_nametime_step/discount:lh
/
_output_shapes
:’’’’’’’’’
5
_user_specified_nametime_step/observation/image:gc
'
_output_shapes
:’’’’’’’’’2
8
_user_specified_name time_step/observation/random_z:hd
'
_output_shapes
:’’’’’’’’’
9
_user_specified_name!time_step/observation/time_step:lh
(
_output_shapes
:’’’’’’’’’
<
_user_specified_name$"policy_state/actor_network_state/0:lh
(
_output_shapes
:’’’’’’’’’
<
_user_specified_name$"policy_state/actor_network_state/1
­

+__inference_function_with_signature_1949068
	step_type

reward
discount
observation_image
observation_random_z
observation_time_step
actor_network_state_0
actor_network_state_1!
unknown:
	unknown_0:
	unknown_1:4

	unknown_2:

	unknown_3:	Ģ 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:

	unknown_9:	

unknown_10:
©

unknown_11:	©
identity	

identity_1

identity_2¢StatefulPartitionedCallš
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_imageobservation_random_zobservation_time_stepactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11* 
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *#
fR
__inference_action_1949035k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Å
_input_shapes³
°:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’2:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:’’’’’’’’’
%
_user_specified_name0/step_type:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
0/reward:OK
#
_output_shapes
:’’’’’’’’’
$
_user_specified_name
0/discount:d`
/
_output_shapes
:’’’’’’’’’
-
_user_specified_name0/observation/image:_[
'
_output_shapes
:’’’’’’’’’2
0
_user_specified_name0/observation/random_z:`\
'
_output_shapes
:’’’’’’’’’
1
_user_specified_name0/observation/time_step:a]
(
_output_shapes
:’’’’’’’’’
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:’’’’’’’’’
1
_user_specified_name1/actor_network_state/1

U
%__inference_get_initial_state_1949881

batch_size
identity

identity_1H
packedPack
batch_size*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’W
IdentityIdentityzeros:output:0*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_1Identityzeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
÷

I__inference_sequential_5_layer_call_and_return_conditional_losses_9449701

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identity¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ģ
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’

dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’

NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Š
e
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449332

inputs
identity¹
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449317i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ę
£
.__inference_sequential_4_layer_call_fn_9449617

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449248p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ģ

*__inference_conv2d_2_layer_call_fn_9449750

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallā
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9448805w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Ļ
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449094

inputs"
dense_12_9449088:4

dense_12_9449090:

identity¢ dense_12/StatefulPartitionedCall¹
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449070
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_9449088dense_12_9449090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449028x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
­
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449222

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

F
*__inference_lambda_5_layer_call_fn_9449809

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449070`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
½
E
)__inference_re_lu_2_layer_call_fn_9449896

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449167h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ž
«
.__inference_sequential_4_layer_call_fn_9449185
lambda_4_input!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449178p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_4_input
Č
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449912

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ģ

.__inference_sequential_5_layer_call_fn_9449558

inputs
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449035o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
š
×
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449248

inputs*
conv2d_2_9449240:
conv2d_2_9449242:
identity¢ conv2d_2/StatefulPartitionedCallĮ
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449222
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_9449240conv2d_2_9449242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449156ā
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449167Ö
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449175r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Č	
ö
E__inference_dense_12_layer_call_and_return_conditional_losses_9449028

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’4
 
_user_specified_nameinputs
Ž
'
%__inference_signature_wrapper_9448779õ
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1949163*(
_construction_contextkEagerRuntime*
_input_shapes 
ś
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449829

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ø
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
­
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_9448793

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
­
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449741

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ø
×
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449481
lambda_5_input"
dense_12_9449475:4

dense_12_9449477:

identity¢ dense_12/StatefulPartitionedCallĮ
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449421
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_9449475dense_12_9449477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449379x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_5_input
ä
£
.__inference_sequential_5_layer_call_fn_9449042
lambda_5_input
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449035o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_5_input
č
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449901

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ś
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449819

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ø
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Š
ź
#__inference_distribution_fn_1949865
	step_type

reward
discount
observation_image
observation_random_z
observation_time_step
actor_network_state_0
actor_network_state_1
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4

}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:

oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	Ģ ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: 
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: 
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:

tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
©n
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	©
identity	

identity_1	

identity_2	

identity_3

identity_4¢gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp¢fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp¢gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp¢fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp¢tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp¢sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp¢tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp¢sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp¢kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp¢jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp¢lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp¢VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp¢UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
ShapeShapediscount*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Z
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    g
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’I
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’F
RankConst*
_output_shapes
: *
dtype0*
value	B :H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :K
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: @
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’b
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :]
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:b
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:’’’’’’’’’
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0*
T0*(
_output_shapes
:’’’’’’’’’

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_1*
T0*(
_output_shapes
:’’’’’’’’’?
Shape_2Shapediscount*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:\
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’K
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :O
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: B
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:g
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’h
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:N
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :c
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : }
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:f
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:’’’’’’’’’

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ņ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:’’’’’’’’’
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :ń
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ņ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimsobservation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :ā
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:’’’’’’’’’ę
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	¼
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’         Ż
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	¶
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’2   Ū
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	¶
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ū
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’¬
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’ø
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¹
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
®
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’µ
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’³
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?“
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4¼
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4°
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
®
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
£
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ū
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’ĢØ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’Ģ
  ą
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ģ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	Ģ *
dtype0ē
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ ņ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0č
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0é
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ ņ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ µ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	·
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¹
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ¹
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask„
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : č
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ü
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:’’’’’’’’’ 
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:©
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¤
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:Ł
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’ Ō
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:¦
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ø
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ø
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¾
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskŖ
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ¹
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:’’’’’’’’’
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ī
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Č
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ņ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ī
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’ü
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 ü
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 ×
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ū
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0ē
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¤
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ź
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’ä
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:’’’’’’’’’
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ķ
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’¦
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¹
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split’
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’Ņ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’ł
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’Õ
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’Ō
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’ö
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ł
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ś
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:’’’’’’’’’ņ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
squeeze_dims
ö
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
©*
dtype0¬
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’©ó
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:©*
dtype0·
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’©
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’©   ”
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’©“
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ö
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:’’’’’’’’’T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0	*
_output_shapes
: µ

Identity_1IdentityhCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:’’’’’’’’’[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0	*
_output_shapes
: °

Identity_3Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’°

Identity_4Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’Į
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Å
_input_shapes³
°:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’2:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : 2Ņ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2Š
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2Ņ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2Š
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2ģ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2ź
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2ģ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2ź
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2Ś
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2Ų
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2Ü
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2°
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2®
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	step_type:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_namereward:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
discount:b^
/
_output_shapes
:’’’’’’’’’
+
_user_specified_nameobservation/image:]Y
'
_output_shapes
:’’’’’’’’’2
.
_user_specified_nameobservation/random_z:^Z
'
_output_shapes
:’’’’’’’’’
/
_user_specified_nameobservation/time_step:_[
(
_output_shapes
:’’’’’’’’’
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:’’’’’’’’’
/
_user_specified_nameactor_network_state/1
ę
£
.__inference_sequential_4_layer_call_fn_9449490

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448827p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
³
G
+__inference_flatten_6_layer_call_fn_9449906

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449175a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
©

ž
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449156

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
æ
R
.__inference_sequential_6_layer_call_fn_9449305
lambda_6_input
identity¼
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449302`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:W S
'
_output_shapes
:’’’’’’’’’2
(
_user_specified_namelambda_6_input
ø

%__inference_signature_wrapper_9448758
discount
observation_image
observation_random_z
observation_time_step

reward
	step_type
actor_network_state_0
actor_network_state_1!
unknown:
	unknown_0:
	unknown_1:4

	unknown_2:

	unknown_3:	Ģ 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:

	unknown_9:	

unknown_10:
©

unknown_11:	©
identity	

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_imageobservation_random_zobservation_time_stepactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11* 
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1949068k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Å
_input_shapes³
°:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’2:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:’’’’’’’’’
$
_user_specified_name
0/discount:d`
/
_output_shapes
:’’’’’’’’’
-
_user_specified_name0/observation/image:_[
'
_output_shapes
:’’’’’’’’’2
0
_user_specified_name0/observation/random_z:`\
'
_output_shapes
:’’’’’’’’’
1
_user_specified_name0/observation/time_step:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
0/reward:PL
#
_output_shapes
:’’’’’’’’’
%
_user_specified_name0/step_type:a]
(
_output_shapes
:’’’’’’’’’
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:’’’’’’’’’
1
_user_specified_name1/actor_network_state/1
ś
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449950

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ø
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ł
e
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449663

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
æ
R
.__inference_sequential_6_layer_call_fn_9448989
lambda_6_input
identity¼
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448981`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:W S
'
_output_shapes
:’’’’’’’’’2
(
_user_specified_namelambda_6_input

U
%__inference_signature_wrapper_9448767

batch_size
identity

identity_1°
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1949136a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’c

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size

ß
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448925
lambda_4_input*
conv2d_2_9448917:
conv2d_2_9448919:
identity¢ conv2d_2/StatefulPartitionedCallÉ
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9448793
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_9448917conv2d_2_9448919*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9448805ā
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9448816Ö
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9448824r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_4_input

Ļ
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449035

inputs"
dense_12_9449029:4

dense_12_9449031:

identity¢ dense_12/StatefulPartitionedCall¹
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449016
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_9449029dense_12_9449031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449028x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
æ
R
.__inference_sequential_6_layer_call_fn_9449340
lambda_6_input
identity¼
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449332`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:W S
'
_output_shapes
:’’’’’’’’’2
(
_user_specified_namelambda_6_input
ś
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449070

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ø
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
÷

I__inference_sequential_5_layer_call_and_return_conditional_losses_9449599

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identity¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ģ
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’

dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’

NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
õ
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449799

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
Č	
ö
E__inference_dense_12_layer_call_and_return_conditional_losses_9449379

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’4
 
_user_specified_nameinputs
ģ

*__inference_conv2d_2_layer_call_fn_9449881

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallā
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449156w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

[
+__inference_function_with_signature_1949136

batch_size
identity

identity_1Ŗ
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_get_initial_state_1949131a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’c

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size

F
*__inference_lambda_5_layer_call_fn_9449935

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449367`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
š
×
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448897

inputs*
conv2d_2_9448889:
conv2d_2_9448891:
identity¢ conv2d_2/StatefulPartitionedCallĮ
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_4_layer_call_and_return_conditional_losses_9448871
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_9448889conv2d_2_9448891*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9448805ā
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9448816Ö
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_9448824r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

¦
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449515

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’W
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ŗ
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’i
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’j
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ą{
Ż
#__inference__traced_restore_9450187
file_prefix&
assignvariableop_global_step:	 >
$assignvariableop_1_conv2d_2_kernel_1:0
"assignvariableop_2_conv2d_2_bias_1:6
$assignvariableop_3_dense_12_kernel_1:4
0
"assignvariableop_4_dense_12_bias_1:

xassignvariableop_5_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel:	Ģ 
vassignvariableop_6_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias: 
xassignvariableop_7_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel:  
vassignvariableop_8_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias: 
passignvariableop_9_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel:	 
{assignvariableop_10_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel:
~
oassignvariableop_11_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_bias:	|
hassignvariableop_12_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel:
©u
fassignvariableop_13_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias:	©=
#assignvariableop_14_conv2d_2_kernel:/
!assignvariableop_15_conv2d_2_bias:5
#assignvariableop_16_dense_12_kernel:4
/
!assignvariableop_17_dense_12_bias:
t
aassignvariableop_18_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernel:	Ģ m
_assignvariableop_19_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_bias: s
aassignvariableop_20_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernel:  m
_assignvariableop_21_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_bias: l
Yassignvariableop_22_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernel:	  v
cassignvariableop_23_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernel:	( f
Wassignvariableop_24_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_bias:	 S
Aassignvariableop_25_adversary_env_valuernnnetwork_dense_17_kernel:(M
?assignvariableop_26_adversary_env_valuernnnetwork_dense_17_bias:
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9š

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

value
B
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_global_stepIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_2_kernel_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_2_bias_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_dense_12_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_12_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ē
AssignVariableOp_5AssignVariableOpxassignvariableop_5_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:å
AssignVariableOp_6AssignVariableOpvassignvariableop_6_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ē
AssignVariableOp_7AssignVariableOpxassignvariableop_7_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:å
AssignVariableOp_8AssignVariableOpvassignvariableop_8_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ß
AssignVariableOp_9AssignVariableOppassignvariableop_9_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ģ
AssignVariableOp_10AssignVariableOp{assignvariableop_10_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ą
AssignVariableOp_11AssignVariableOpoassignvariableop_11_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_12AssignVariableOphassignvariableop_12_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_13AssignVariableOpfassignvariableop_13_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_12_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_12_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ņ
AssignVariableOp_18AssignVariableOpaassignvariableop_18_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_19AssignVariableOp_assignvariableop_19_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ņ
AssignVariableOp_20AssignVariableOpaassignvariableop_20_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_21AssignVariableOp_assignvariableop_21_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ź
AssignVariableOp_22AssignVariableOpYassignvariableop_22_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ō
AssignVariableOp_23AssignVariableOpcassignvariableop_23_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Č
AssignVariableOp_24AssignVariableOpWassignvariableop_24_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_25AssignVariableOpAassignvariableop_25_adversary_env_valuernnnetwork_dense_17_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_26AssignVariableOp?assignvariableop_26_adversary_env_valuernnnetwork_dense_17_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ”
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
õ
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449317

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
č
m
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449350
lambda_6_input
identityĮ
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449317i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:W S
'
_output_shapes
:’’’’’’’’’2
(
_user_specified_namelambda_6_input
ś
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449960

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ø
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ł
e
%__inference_signature_wrapper_9448775
unknown:	 
identity	¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_1949152^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
Č	
ö
E__inference_dense_12_layer_call_and_return_conditional_losses_9449848

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’4
 
_user_specified_nameinputs

Ļ
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449386

inputs"
dense_12_9449380:4

dense_12_9449382:

identity¢ dense_12/StatefulPartitionedCall¹
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449367
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_9449380dense_12_9449382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_9449379x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
­
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449734

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ł
e
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449549

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
Ķ
k
+__inference_function_with_signature_1949152
unknown:	 
identity	¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_<lambda>_3489^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
ę
£
.__inference_sequential_4_layer_call_fn_9449499

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448897p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
č
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449167

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
­
a
E__inference_lambda_4_layer_call_and_return_conditional_losses_9448871

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:’’’’’’’’’N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
÷

I__inference_sequential_5_layer_call_and_return_conditional_losses_9449717

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identity¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4Ģ
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:’’’’’’’’’4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’4   
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’4
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’

dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’

NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Č
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_9448824

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
õ
a
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449299

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’2:O K
'
_output_shapes
:’’’’’’’’’2
 
_user_specified_nameinputs
ž
«
.__inference_sequential_4_layer_call_fn_9449264
lambda_4_input!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallē
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449248p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:’’’’’’’’’
(
_user_specified_namelambda_4_input
©

ž
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449760

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
©

ž
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449891

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ģ

.__inference_sequential_5_layer_call_fn_9449567

inputs
unknown:4

	unknown_0:

identity¢StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"æL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ē
action¼
4

0/discount&
action_0_discount:0’’’’’’’’’
R
0/observation/image;
action_0_observation_image:0’’’’’’’’’
P
0/observation/random_z6
action_0_observation_random_z:0’’’’’’’’’2
R
0/observation/time_step7
 action_0_observation_time_step:0’’’’’’’’’
0
0/reward$
action_0_reward:0’’’’’’’’’
6
0/step_type'
action_0_step_type:0’’’’’’’’’
S
1/actor_network_state/08
 action_1_actor_network_state_0:0’’’’’’’’’
S
1/actor_network_state/18
 action_1_actor_network_state_1:0’’’’’’’’’6
action,
StatefulPartitionedCall:0	’’’’’’’’’P
state/actor_network_state/01
StatefulPartitionedCall:1’’’’’’’’’P
state/actor_network_state/11
StatefulPartitionedCall:2’’’’’’’’’tensorflow/serving/predict*ī
get_initial_stateŲ
2

batch_size$
get_initial_state_batch_size:0 B
actor_network_state/0)
PartitionedCall:0’’’’’’’’’B
actor_network_state/1)
PartitionedCall:1’’’’’’’’’tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:ąĻ
ū
collect_data_spec
policy_state_spec

train_step
metadata
model_variables
_all_assets

action
distribution
	get_initial_state

get_metadata
get_train_step

signatures"
_generic_user_object
9
observation
1"
trackable_tuple_wrapper
9
actor_network_state"
trackable_dict_wrapper
:	 (2global_step
 "
trackable_dict_wrapper
ē
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
'24
(25"
trackable_tuple_wrapper

)_time_step_spec
*_policy_state_spec
+_policy_step_spec
,_trajectory_spec
-_wrapped_policy"
trackable_dict_wrapper
¹
.trace_0
/trace_12
__inference_action_1949408
__inference_action_1949649Ē
¾²ŗ
FullArgSpec8
args0-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults¢	
¢ 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z.trace_0z/trace_1
š
0trace_02Ó
#__inference_distribution_fn_1949865«
¤² 
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z0trace_0
ķ
1trace_02Š
%__inference_get_initial_state_1949881¦
²
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z1trace_0
ÆB¬
__inference_<lambda>_3492"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ÆB¬
__inference_<lambda>_3489"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
`

2action
3get_initial_state
4get_train_step
5get_metadata"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
):'2conv2d_2/kernel
:2conv2d_2/bias
!:4
2dense_12/kernel
:
2dense_12/bias
x:v	Ģ 2eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel
q:o 2cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias
w:u  2eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel
q:o 2cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias
p:n	 2]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel
{:y
2gadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel
j:h2[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias
h:f
©2Tadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
a:_©2Radversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
):'2conv2d_2/kernel
:2conv2d_2/bias
!:4
2dense_12/kernel
:
2dense_12/bias
`:^	Ģ 2Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel
Y:W 2Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias
_:]  2Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel
Y:W 2Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias
X:V	  2Eadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel
b:`	( 2Oadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel
R:P 2Cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias
?:=(2-adversary_env/ValueRnnNetwork/dense_17/kernel
9:72+adversary_env/ValueRnnNetwork/dense_17/bias
9
observation
3"
trackable_tuple_wrapper
9
actor_network_state"
trackable_dict_wrapper
3
	*state
*1"
trackable_tuple_wrapper
9
observation
1"
trackable_tuple_wrapper
 
6_actor_network
7_time_step_spec
8_policy_state_spec
9_policy_step_spec
:_trajectory_spec
;_value_network"
_generic_user_object
öBó
__inference_action_1949408	step_typerewarddiscountobservation/imageobservation/random_zobservation/time_stepactor_network_state/0actor_network_state/1"Ē
¾²ŗ
FullArgSpec8
args0-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults¢	
¢ 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ĢBÉ
__inference_action_1949649time_step/step_typetime_step/rewardtime_step/discounttime_step/observation/imagetime_step/observation/random_ztime_step/observation/time_step"policy_state/actor_network_state/0"policy_state/actor_network_state/1"Ē
¾²ŗ
FullArgSpec8
args0-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults¢	
¢ 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ćBą
#__inference_distribution_fn_1949865	step_typerewarddiscountobservation/imageobservation/random_zobservation/time_stepactor_network_state/0actor_network_state/1"«
¤² 
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
įBŽ
%__inference_get_initial_state_1949881
batch_size"¦
²
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ÜBŁ
%__inference_signature_wrapper_9448758
0/discount0/observation/image0/observation/random_z0/observation/time_step0/reward0/step_type1/actor_network_state/01/actor_network_state/1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ĻBĢ
%__inference_signature_wrapper_9448767
batch_size"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ĮB¾
%__inference_signature_wrapper_9448775"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ĮB¾
%__inference_signature_wrapper_9448779"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ū
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_input_tensor_spec
C_state_spec
D_lstm_encoder
E_projection_networks"
_tf_keras_layer
9
Fobservation
F3"
trackable_tuple_wrapper
9
Gactor_network_state"
trackable_dict_wrapper
3
	8state
81"
trackable_tuple_wrapper
9
Fobservation
F1"
trackable_tuple_wrapper
ż
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_input_tensor_spec
O_state_spec
P_lstm_encoder
Q_postprocessing_layers"
_tf_keras_layer
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
ā2ßÜ
Ó²Ļ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ā2ßÜ
Ó²Ļ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper

W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_input_tensor_spec
^_state_spec
__input_encoder
`_lstm_network
a_output_encoder"
_tf_keras_layer
¼
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_projection_layer"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12"
trackable_list_wrapper
~
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12"
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper

n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_input_tensor_spec
u_state_spec
v_input_encoder
w_lstm_network
x_output_encoder"
_tf_keras_layer
»
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
±
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
ā2ßÜ
Ó²Ļ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ā2ßÜ
Ó²Ļ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ŗ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_input_tensor_spec
_preprocessing_nest
_flat_preprocessing_layers
_preprocessing_combiner
_postprocessing_layers"
_tf_keras_layer
¶
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	cell"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
Ō2ŃĪ
Å²Į
FullArgSpec?
args74
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ō2ŃĪ
Å²Į
FullArgSpec?
args74
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Į
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
 4
!5
"6
#7
$8
%9
&10"
trackable_list_wrapper
n
0
1
2
3
 4
!5
"6
#7
$8
%9
&10"
trackable_list_wrapper
 "
trackable_list_wrapper
²
”non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
„layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
ā2ßÜ
Ó²Ļ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ā2ßÜ
Ó²Ļ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ŗ
¦	variables
§trainable_variables
Øregularization_losses
©	keras_api
Ŗ__call__
+«&call_and_return_all_conditional_losses
¬_input_tensor_spec
­_preprocessing_nest
®_flat_preprocessing_layers
Æ_preprocessing_combiner
°_postprocessing_layers"
_tf_keras_layer
¶
±	variables
²trainable_variables
³regularization_losses
“	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses
	·cell"
_tf_keras_layer
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ønon_trainable_variables
¹layers
ŗmetrics
 »layer_regularization_losses
¼layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
½non_trainable_variables
¾layers
æmetrics
 Ąlayer_regularization_losses
Įlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
8
Ā0
Ć1
Ä2"
trackable_list_wrapper
«
Å	variables
Ętrainable_variables
Ēregularization_losses
Č	keras_api
É__call__
+Ź&call_and_return_all_conditional_losses"
_tf_keras_layer
8
Ė0
Ģ1
Ķ2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Īnon_trainable_variables
Ļlayers
Šmetrics
 Ńlayer_regularization_losses
Ņlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
į2ŽŪ
Ņ²Ī
FullArgSpecH
args@=
jself
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
į2ŽŪ
Ņ²Ī
FullArgSpecH
args@=
jself
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 

Ó	variables
Ōtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ų&call_and_return_all_conditional_losses
Ł_random_generator
Ś
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ūnon_trainable_variables
Ülayers
Żmetrics
 Žlayer_regularization_losses
ßlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
 4
!5
"6
#7"
trackable_list_wrapper
X
0
1
2
3
 4
!5
"6
#7"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ąnon_trainable_variables
įlayers
āmetrics
 ćlayer_regularization_losses
älayer_metrics
¦	variables
§trainable_variables
Øregularization_losses
Ŗ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
8
å0
ę1
ē2"
trackable_list_wrapper
«
č	variables
étrainable_variables
źregularization_losses
ė	keras_api
ģ__call__
+ķ&call_and_return_all_conditional_losses"
_tf_keras_layer
8
ī0
ļ1
š2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ńnon_trainable_variables
ņlayers
ómetrics
 ōlayer_regularization_losses
õlayer_metrics
±	variables
²trainable_variables
³regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
į2ŽŪ
Ņ²Ī
FullArgSpecH
args@=
jself
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
į2ŽŪ
Ņ²Ī
FullArgSpecH
args@=
jself
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 

ö	variables
÷trainable_variables
ųregularization_losses
ł	keras_api
ś__call__
+ū&call_and_return_all_conditional_losses
ü_random_generator
ż
state_size

$kernel
%recurrent_kernel
&bias"
_tf_keras_layer
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
X
Ā0
Ć1
Ä2
3
Ė4
Ģ5
Ķ6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

žlayer-0
’layer_with_weights-0
’layer-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
¾
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
ē
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Å	variables
Ętrainable_variables
Ēregularization_losses
É__call__
+Ź&call_and_return_all_conditional_losses
'Ź"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
«
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+”&call_and_return_all_conditional_losses"
_tf_keras_layer
Į
¢	variables
£trainable_variables
¤regularization_losses
„	keras_api
¦__call__
+§&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Į
Ø	variables
©trainable_variables
Ŗregularization_losses
«	keras_api
¬__call__
+­&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
®non_trainable_variables
Ælayers
°metrics
 ±layer_regularization_losses
²layer_metrics
Ó	variables
Ōtrainable_variables
Õregularization_losses
×__call__
+Ų&call_and_return_all_conditional_losses
'Ų"call_and_return_conditional_losses"
_generic_user_object
Ä2Į¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ä2Į¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
"
_generic_user_object
 "
trackable_list_wrapper
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
X
å0
ę1
ē2
Æ3
ī4
ļ5
š6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

³layer-0
“layer_with_weights-0
“layer-1
µlayer-2
¶layer-3
·	variables
øtrainable_variables
¹regularization_losses
ŗ	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"
_tf_keras_sequential
¾
½layer-0
¾	variables
ætrainable_variables
Ąregularization_losses
Į	keras_api
Ā__call__
+Ć&call_and_return_all_conditional_losses"
_tf_keras_sequential
ē
Älayer-0
Ålayer_with_weights-0
Ålayer-1
Ę	variables
Ētrainable_variables
Čregularization_losses
É	keras_api
Ź__call__
+Ė&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ģnon_trainable_variables
Ķlayers
Īmetrics
 Ļlayer_regularization_losses
Šlayer_metrics
č	variables
étrainable_variables
źregularization_losses
ģ__call__
+ķ&call_and_return_all_conditional_losses
'ķ"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
«
Ń	variables
Ņtrainable_variables
Óregularization_losses
Ō	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
Į
×	variables
Ųtrainable_variables
Łregularization_losses
Ś	keras_api
Ū__call__
+Ü&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
Į
Ż	variables
Žtrainable_variables
ßregularization_losses
ą	keras_api
į__call__
+ā&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
 "
trackable_list_wrapper
(
·0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ćnon_trainable_variables
älayers
åmetrics
 ęlayer_regularization_losses
ēlayer_metrics
ö	variables
÷trainable_variables
ųregularization_losses
ś__call__
+ū&call_and_return_all_conditional_losses
'ū"call_and_return_conditional_losses"
_generic_user_object
Ä2Į¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ä2Į¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
"
_generic_user_object
 "
trackable_list_wrapper
«
č	variables
étrainable_variables
źregularization_losses
ė	keras_api
ģ__call__
+ķ&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
ī	variables
ļtrainable_variables
šregularization_losses
ń	keras_api
ņ__call__
+ó&call_and_return_all_conditional_losses

kernel
bias
!ō_jit_compiled_convolution_op"
_tf_keras_layer
«
õ	variables
ötrainable_variables
÷regularization_losses
ų	keras_api
ł__call__
+ś&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ū	variables
ütrainable_variables
żregularization_losses
ž	keras_api
’__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ö
trace_0
trace_1
trace_2
trace_32
.__inference_sequential_4_layer_call_fn_9448834
.__inference_sequential_4_layer_call_fn_9449490
.__inference_sequential_4_layer_call_fn_9449499
.__inference_sequential_4_layer_call_fn_9448913Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ā
trace_0
trace_1
trace_2
trace_32ļ
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449515
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449531
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448925
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448937Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ö
trace_0
trace_1
trace_2
trace_32
.__inference_sequential_6_layer_call_fn_9448954
.__inference_sequential_6_layer_call_fn_9449536
.__inference_sequential_6_layer_call_fn_9449541
.__inference_sequential_6_layer_call_fn_9448989Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ā
trace_0
trace_1
trace_2
 trace_32ļ
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449545
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449549
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448994
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448999Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2z trace_3
«
”	variables
¢trainable_variables
£regularization_losses
¤	keras_api
„__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
Į
§	variables
Øtrainable_variables
©regularization_losses
Ŗ	keras_api
«__call__
+¬&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
­non_trainable_variables
®layers
Æmetrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ö
²trace_0
³trace_1
“trace_2
µtrace_32
.__inference_sequential_5_layer_call_fn_9449042
.__inference_sequential_5_layer_call_fn_9449558
.__inference_sequential_5_layer_call_fn_9449567
.__inference_sequential_5_layer_call_fn_9449110Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 z²trace_0z³trace_1z“trace_2zµtrace_3
ā
¶trace_0
·trace_1
øtrace_2
¹trace_32ļ
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449583
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449599
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449120
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449130Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 z¶trace_0z·trace_1zøtrace_2z¹trace_3
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
ø
ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ænon_trainable_variables
Ąlayers
Įmetrics
 Ālayer_regularization_losses
Ćlayer_metrics
¢	variables
£trainable_variables
¤regularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Änon_trainable_variables
Ålayers
Ęmetrics
 Ēlayer_regularization_losses
Člayer_metrics
Ø	variables
©trainable_variables
Ŗregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
«
É	variables
Źtrainable_variables
Ėregularization_losses
Ģ	keras_api
Ķ__call__
+Ī&call_and_return_all_conditional_losses"
_tf_keras_layer
ä
Ļ	variables
Štrainable_variables
Ńregularization_losses
Ņ	keras_api
Ó__call__
+Ō&call_and_return_all_conditional_losses

kernel
bias
!Õ_jit_compiled_convolution_op"
_tf_keras_layer
«
Ö	variables
×trainable_variables
Ųregularization_losses
Ł	keras_api
Ś__call__
+Ū&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ü	variables
Żtrainable_variables
Žregularization_losses
ß	keras_api
ą__call__
+į&call_and_return_all_conditional_losses"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ānon_trainable_variables
ćlayers
ämetrics
 ålayer_regularization_losses
ęlayer_metrics
·	variables
øtrainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
ö
ētrace_0
čtrace_1
étrace_2
źtrace_32
.__inference_sequential_4_layer_call_fn_9449185
.__inference_sequential_4_layer_call_fn_9449608
.__inference_sequential_4_layer_call_fn_9449617
.__inference_sequential_4_layer_call_fn_9449264Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zētrace_0zčtrace_1zétrace_2zźtrace_3
ā
ėtrace_0
ģtrace_1
ķtrace_2
ītrace_32ļ
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449633
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449649
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449276
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449288Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zėtrace_0zģtrace_1zķtrace_2zītrace_3
«
ļ	variables
štrainable_variables
ńregularization_losses
ņ	keras_api
ó__call__
+ō&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
õnon_trainable_variables
ölayers
÷metrics
 ųlayer_regularization_losses
łlayer_metrics
¾	variables
ætrainable_variables
Ąregularization_losses
Ā__call__
+Ć&call_and_return_all_conditional_losses
'Ć"call_and_return_conditional_losses"
_generic_user_object
ö
śtrace_0
ūtrace_1
ütrace_2
żtrace_32
.__inference_sequential_6_layer_call_fn_9449305
.__inference_sequential_6_layer_call_fn_9449654
.__inference_sequential_6_layer_call_fn_9449659
.__inference_sequential_6_layer_call_fn_9449340Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zśtrace_0zūtrace_1zütrace_2zżtrace_3
ā
žtrace_0
’trace_1
trace_2
trace_32ļ
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449663
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449667
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449345
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449350Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zžtrace_0z’trace_1ztrace_2ztrace_3
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Į
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ę	variables
Ētrainable_variables
Čregularization_losses
Ź__call__
+Ė&call_and_return_all_conditional_losses
'Ė"call_and_return_conditional_losses"
_generic_user_object
ö
trace_0
trace_1
trace_2
trace_32
.__inference_sequential_5_layer_call_fn_9449393
.__inference_sequential_5_layer_call_fn_9449676
.__inference_sequential_5_layer_call_fn_9449685
.__inference_sequential_5_layer_call_fn_9449461Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ā
trace_0
trace_1
trace_2
trace_32ļ
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449701
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449717
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449471
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449481Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
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
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ń	variables
Ņtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
 non_trainable_variables
”layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
×	variables
Ųtrainable_variables
Łregularization_losses
Ū__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
„non_trainable_variables
¦layers
§metrics
 Ølayer_regularization_losses
©layer_metrics
Ż	variables
Žtrainable_variables
ßregularization_losses
į__call__
+ā&call_and_return_all_conditional_losses
'ā"call_and_return_conditional_losses"
_generic_user_object
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
ø
Ŗnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
č	variables
étrainable_variables
źregularization_losses
ģ__call__
+ķ&call_and_return_all_conditional_losses
'ķ"call_and_return_conditional_losses"
_generic_user_object
Ö
Ætrace_0
°trace_12
*__inference_lambda_4_layer_call_fn_9449722
*__inference_lambda_4_layer_call_fn_9449727Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zÆtrace_0z°trace_1

±trace_0
²trace_12Ń
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449734
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449741Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 z±trace_0z²trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
³non_trainable_variables
“layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
ī	variables
ļtrainable_variables
šregularization_losses
ņ__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
š
øtrace_02Ń
*__inference_conv2d_2_layer_call_fn_9449750¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zøtrace_0

¹trace_02ģ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449760¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¹trace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ł__call__
+ś&call_and_return_all_conditional_losses
'ś"call_and_return_conditional_losses"
_generic_user_object
ļ
ætrace_02Š
)__inference_re_lu_2_layer_call_fn_9449765¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zætrace_0

Ątrace_02ė
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449770¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĄtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Įnon_trainable_variables
Ālayers
Ćmetrics
 Älayer_regularization_losses
Ålayer_metrics
ū	variables
ütrainable_variables
żregularization_losses
’__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ń
Ętrace_02Ņ
+__inference_flatten_6_layer_call_fn_9449775¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĘtrace_0

Ētrace_02ķ
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449781¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĒtrace_0
 "
trackable_list_wrapper
@
ž0
’1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_4_layer_call_fn_9448834lambda_4_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_4_layer_call_fn_9449490inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_4_layer_call_fn_9449499inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
.__inference_sequential_4_layer_call_fn_9448913lambda_4_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449515inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449531inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448925lambda_4_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448937lambda_4_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Čnon_trainable_variables
Élayers
Źmetrics
 Ėlayer_regularization_losses
Ģlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö
Ķtrace_0
Ītrace_12
*__inference_lambda_6_layer_call_fn_9449786
*__inference_lambda_6_layer_call_fn_9449791Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zĶtrace_0zĪtrace_1

Ļtrace_0
Štrace_12Ń
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449795
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449799Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zĻtrace_0zŠtrace_1
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_6_layer_call_fn_9448954lambda_6_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_6_layer_call_fn_9449536inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_6_layer_call_fn_9449541inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
.__inference_sequential_6_layer_call_fn_9448989lambda_6_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449545inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449549inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448994lambda_6_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448999lambda_6_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ńnon_trainable_variables
Ņlayers
Ómetrics
 Ōlayer_regularization_losses
Õlayer_metrics
”	variables
¢trainable_variables
£regularization_losses
„__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
Ö
Ötrace_0
×trace_12
*__inference_lambda_5_layer_call_fn_9449804
*__inference_lambda_5_layer_call_fn_9449809Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zÖtrace_0z×trace_1

Ųtrace_0
Łtrace_12Ń
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449819
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449829Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zŲtrace_0zŁtrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Śnon_trainable_variables
Ūlayers
Ümetrics
 Żlayer_regularization_losses
Žlayer_metrics
§	variables
Øtrainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
š
ßtrace_02Ń
*__inference_dense_12_layer_call_fn_9449838¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zßtrace_0

ątrace_02ģ
E__inference_dense_12_layer_call_and_return_conditional_losses_9449848¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zątrace_0
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_5_layer_call_fn_9449042lambda_5_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_5_layer_call_fn_9449558inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_5_layer_call_fn_9449567inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
.__inference_sequential_5_layer_call_fn_9449110lambda_5_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449583inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449599inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449120lambda_5_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449130lambda_5_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
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
ø
įnon_trainable_variables
ālayers
ćmetrics
 älayer_regularization_losses
ålayer_metrics
É	variables
Źtrainable_variables
Ėregularization_losses
Ķ__call__
+Ī&call_and_return_all_conditional_losses
'Ī"call_and_return_conditional_losses"
_generic_user_object
Ö
ętrace_0
ētrace_12
*__inference_lambda_4_layer_call_fn_9449853
*__inference_lambda_4_layer_call_fn_9449858Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zętrace_0zētrace_1

čtrace_0
étrace_12Ń
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449865
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449872Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 zčtrace_0zétrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
źnon_trainable_variables
ėlayers
ģmetrics
 ķlayer_regularization_losses
īlayer_metrics
Ļ	variables
Štrainable_variables
Ńregularization_losses
Ó__call__
+Ō&call_and_return_all_conditional_losses
'Ō"call_and_return_conditional_losses"
_generic_user_object
š
ļtrace_02Ń
*__inference_conv2d_2_layer_call_fn_9449881¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zļtrace_0

štrace_02ģ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449891¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zštrace_0
“2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ńnon_trainable_variables
ņlayers
ómetrics
 ōlayer_regularization_losses
õlayer_metrics
Ö	variables
×trainable_variables
Ųregularization_losses
Ś__call__
+Ū&call_and_return_all_conditional_losses
'Ū"call_and_return_conditional_losses"
_generic_user_object
ļ
ötrace_02Š
)__inference_re_lu_2_layer_call_fn_9449896¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zötrace_0

÷trace_02ė
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449901¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z÷trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ųnon_trainable_variables
łlayers
śmetrics
 ūlayer_regularization_losses
ülayer_metrics
Ü	variables
Żtrainable_variables
Žregularization_losses
ą__call__
+į&call_and_return_all_conditional_losses
'į"call_and_return_conditional_losses"
_generic_user_object
ń
żtrace_02Ņ
+__inference_flatten_6_layer_call_fn_9449906¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zżtrace_0

žtrace_02ķ
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449912¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zžtrace_0
 "
trackable_list_wrapper
@
³0
“1
µ2
¶3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_4_layer_call_fn_9449185lambda_4_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_4_layer_call_fn_9449608inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_4_layer_call_fn_9449617inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
.__inference_sequential_4_layer_call_fn_9449264lambda_4_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449633inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449649inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449276lambda_4_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449288lambda_4_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
’non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ļ	variables
štrainable_variables
ńregularization_losses
ó__call__
+ō&call_and_return_all_conditional_losses
'ō"call_and_return_conditional_losses"
_generic_user_object
Ö
trace_0
trace_12
*__inference_lambda_6_layer_call_fn_9449917
*__inference_lambda_6_layer_call_fn_9449922Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1

trace_0
trace_12Ń
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449926
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449930Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
(
½0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_6_layer_call_fn_9449305lambda_6_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_6_layer_call_fn_9449654inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_6_layer_call_fn_9449659inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
.__inference_sequential_6_layer_call_fn_9449340lambda_6_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449663inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449667inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449345lambda_6_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449350lambda_6_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö
trace_0
trace_12
*__inference_lambda_5_layer_call_fn_9449935
*__inference_lambda_5_layer_call_fn_9449940Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1

trace_0
trace_12Ń
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449950
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449960Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 ztrace_0ztrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
š
trace_02Ń
*__inference_dense_12_layer_call_fn_9449969¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0

trace_02ģ
E__inference_dense_12_layer_call_and_return_conditional_losses_9449979¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0
 "
trackable_list_wrapper
0
Ä0
Å1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_5_layer_call_fn_9449393lambda_5_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_5_layer_call_fn_9449676inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Bż
.__inference_sequential_5_layer_call_fn_9449685inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
.__inference_sequential_5_layer_call_fn_9449461lambda_5_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449701inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449717inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449471lambda_5_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
£B 
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449481lambda_5_input"Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
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
üBł
*__inference_lambda_4_layer_call_fn_9449722inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
üBł
*__inference_lambda_4_layer_call_fn_9449727inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449734inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449741inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
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
ŽBŪ
*__inference_conv2d_2_layer_call_fn_9449750inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449760inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
ŻBŚ
)__inference_re_lu_2_layer_call_fn_9449765inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ųBõ
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449770inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
ßBÜ
+__inference_flatten_6_layer_call_fn_9449775inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449781inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
üBł
*__inference_lambda_6_layer_call_fn_9449786inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
üBł
*__inference_lambda_6_layer_call_fn_9449791inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449795inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449799inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
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
üBł
*__inference_lambda_5_layer_call_fn_9449804inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
üBł
*__inference_lambda_5_layer_call_fn_9449809inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449819inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449829inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
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
ŽBŪ
*__inference_dense_12_layer_call_fn_9449838inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_dense_12_layer_call_and_return_conditional_losses_9449848inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
üBł
*__inference_lambda_4_layer_call_fn_9449853inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
üBł
*__inference_lambda_4_layer_call_fn_9449858inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449865inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449872inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
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
ŽBŪ
*__inference_conv2d_2_layer_call_fn_9449881inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449891inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
ŻBŚ
)__inference_re_lu_2_layer_call_fn_9449896inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ųBõ
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449901inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
ßBÜ
+__inference_flatten_6_layer_call_fn_9449906inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
śB÷
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449912inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
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
üBł
*__inference_lambda_6_layer_call_fn_9449917inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
üBł
*__inference_lambda_6_layer_call_fn_9449922inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449926inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449930inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
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
üBł
*__inference_lambda_5_layer_call_fn_9449935inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
üBł
*__inference_lambda_5_layer_call_fn_9449940inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449950inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
B
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449960inputs"Ą
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
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
ŽBŪ
*__inference_dense_12_layer_call_fn_9449969inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
E__inference_dense_12_layer_call_and_return_conditional_losses_9449979inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 8
__inference_<lambda>_3489¢

¢ 
Ŗ " 	1
__inference_<lambda>_3492¢

¢ 
Ŗ "Ŗ 
__inference_action_1949408õż¢ł
ń¢ķ
ß²Ū
TimeStep,
	step_type
	step_type’’’’’’’’’&
reward
reward’’’’’’’’’*
discount
discount’’’’’’’’’Ģ
observation¼Ŗø
<
image30
observation/image’’’’’’’’’
:
random_z.+
observation/random_z’’’’’’’’’2
<
	time_step/,
observation/time_step’’’’’’’’’
Ŗ
~
actor_network_stategd
0-
actor_network_state/0’’’’’’’’’
0-
actor_network_state/1’’’’’’’’’

 
Ŗ "ć²ß

PolicyStep&
action
action’’’’’’’’’	
stateŖ

actor_network_statesp
63
state/actor_network_state/0’’’’’’’’’
63
state/actor_network_state/1’’’’’’’’’
info¢ ģ
__inference_action_1949649ĶÕ¢Ń
É¢Å
²
TimeStep6
	step_type)&
time_step/step_type’’’’’’’’’0
reward&#
time_step/reward’’’’’’’’’4
discount(%
time_step/discount’’’’’’’’’ź
observationŚŖÖ
F
image=:
time_step/observation/image’’’’’’’’’
D
random_z85
time_step/observation/random_z’’’’’’’’’2
F
	time_step96
time_step/observation/time_step’’’’’’’’’
 Ŗ

actor_network_state~
=:
"policy_state/actor_network_state/0’’’’’’’’’
=:
"policy_state/actor_network_state/1’’’’’’’’’

 
Ŗ "ć²ß

PolicyStep&
action
action’’’’’’’’’	
stateŖ

actor_network_statesp
63
state/actor_network_state/0’’’’’’’’’
63
state/actor_network_state/1’’’’’’’’’
info¢ µ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449760l7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "-¢*
# 
0’’’’’’’’’
 µ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9449891l7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "-¢*
# 
0’’’’’’’’’
 
*__inference_conv2d_2_layer_call_fn_9449750_7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ " ’’’’’’’’’
*__inference_conv2d_2_layer_call_fn_9449881_7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ " ’’’’’’’’’„
E__inference_dense_12_layer_call_and_return_conditional_losses_9449848\/¢,
%¢"
 
inputs’’’’’’’’’4
Ŗ "%¢"

0’’’’’’’’’

 „
E__inference_dense_12_layer_call_and_return_conditional_losses_9449979\/¢,
%¢"
 
inputs’’’’’’’’’4
Ŗ "%¢"

0’’’’’’’’’

 }
*__inference_dense_12_layer_call_fn_9449838O/¢,
%¢"
 
inputs’’’’’’’’’4
Ŗ "’’’’’’’’’
}
*__inference_dense_12_layer_call_fn_9449969O/¢,
%¢"
 
inputs’’’’’’’’’4
Ŗ "’’’’’’’’’
ō
#__inference_distribution_fn_1949865Ģł¢õ
ķ¢é
ß²Ū
TimeStep,
	step_type
	step_type’’’’’’’’’&
reward
reward’’’’’’’’’*
discount
discount’’’’’’’’’Ģ
observation¼Ŗø
<
image30
observation/image’’’’’’’’’
:
random_z.+
observation/random_z’’’’’’’’’2
<
	time_step/,
observation/time_step’’’’’’’’’
Ŗ
~
actor_network_stategd
0-
actor_network_state/0’’’’’’’’’
0-
actor_network_state/1’’’’’’’’’
Ŗ "¾²ŗ

PolicyStep
actionõń½¢¹
`
BŖ?

atol 	

loc’’’’’’’’’	

rtol 	
JŖG

allow_nan_statsp

namejDeterministic_1

validate_argsp 
¢
j
parameters
¢ 
¢
jname+tfp.distributions.Deterministic_ACTTypeSpec 
stateŖ

actor_network_statesp
63
state/actor_network_state/0’’’’’’’’’
63
state/actor_network_state/1’’’’’’’’’
info¢ «
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449781a7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 «
F__inference_flatten_6_layer_call_and_return_conditional_losses_9449912a7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
+__inference_flatten_6_layer_call_fn_9449775T7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
+__inference_flatten_6_layer_call_fn_9449906T7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "’’’’’’’’’Õ
%__inference_get_initial_state_1949881«"¢
¢


batch_size 
Ŗ "Ŗ
~
actor_network_stategd
0-
actor_network_state/0’’’’’’’’’
0-
actor_network_state/1’’’’’’’’’¹
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449734p?¢<
5¢2
(%
inputs’’’’’’’’’

 
p 
Ŗ "-¢*
# 
0’’’’’’’’’
 ¹
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449741p?¢<
5¢2
(%
inputs’’’’’’’’’

 
p
Ŗ "-¢*
# 
0’’’’’’’’’
 ¹
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449865p?¢<
5¢2
(%
inputs’’’’’’’’’

 
p 
Ŗ "-¢*
# 
0’’’’’’’’’
 ¹
E__inference_lambda_4_layer_call_and_return_conditional_losses_9449872p?¢<
5¢2
(%
inputs’’’’’’’’’

 
p
Ŗ "-¢*
# 
0’’’’’’’’’
 
*__inference_lambda_4_layer_call_fn_9449722c?¢<
5¢2
(%
inputs’’’’’’’’’

 
p 
Ŗ " ’’’’’’’’’
*__inference_lambda_4_layer_call_fn_9449727c?¢<
5¢2
(%
inputs’’’’’’’’’

 
p
Ŗ " ’’’’’’’’’
*__inference_lambda_4_layer_call_fn_9449853c?¢<
5¢2
(%
inputs’’’’’’’’’

 
p 
Ŗ " ’’’’’’’’’
*__inference_lambda_4_layer_call_fn_9449858c?¢<
5¢2
(%
inputs’’’’’’’’’

 
p
Ŗ " ’’’’’’’’’©
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449819`7¢4
-¢*
 
inputs’’’’’’’’’

 
p 
Ŗ "%¢"

0’’’’’’’’’4
 ©
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449829`7¢4
-¢*
 
inputs’’’’’’’’’

 
p
Ŗ "%¢"

0’’’’’’’’’4
 ©
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449950`7¢4
-¢*
 
inputs’’’’’’’’’

 
p 
Ŗ "%¢"

0’’’’’’’’’4
 ©
E__inference_lambda_5_layer_call_and_return_conditional_losses_9449960`7¢4
-¢*
 
inputs’’’’’’’’’

 
p
Ŗ "%¢"

0’’’’’’’’’4
 
*__inference_lambda_5_layer_call_fn_9449804S7¢4
-¢*
 
inputs’’’’’’’’’

 
p 
Ŗ "’’’’’’’’’4
*__inference_lambda_5_layer_call_fn_9449809S7¢4
-¢*
 
inputs’’’’’’’’’

 
p
Ŗ "’’’’’’’’’4
*__inference_lambda_5_layer_call_fn_9449935S7¢4
-¢*
 
inputs’’’’’’’’’

 
p 
Ŗ "’’’’’’’’’4
*__inference_lambda_5_layer_call_fn_9449940S7¢4
-¢*
 
inputs’’’’’’’’’

 
p
Ŗ "’’’’’’’’’4©
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449795`7¢4
-¢*
 
inputs’’’’’’’’’2

 
p 
Ŗ "%¢"

0’’’’’’’’’2
 ©
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449799`7¢4
-¢*
 
inputs’’’’’’’’’2

 
p
Ŗ "%¢"

0’’’’’’’’’2
 ©
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449926`7¢4
-¢*
 
inputs’’’’’’’’’2

 
p 
Ŗ "%¢"

0’’’’’’’’’2
 ©
E__inference_lambda_6_layer_call_and_return_conditional_losses_9449930`7¢4
-¢*
 
inputs’’’’’’’’’2

 
p
Ŗ "%¢"

0’’’’’’’’’2
 
*__inference_lambda_6_layer_call_fn_9449786S7¢4
-¢*
 
inputs’’’’’’’’’2

 
p 
Ŗ "’’’’’’’’’2
*__inference_lambda_6_layer_call_fn_9449791S7¢4
-¢*
 
inputs’’’’’’’’’2

 
p
Ŗ "’’’’’’’’’2
*__inference_lambda_6_layer_call_fn_9449917S7¢4
-¢*
 
inputs’’’’’’’’’2

 
p 
Ŗ "’’’’’’’’’2
*__inference_lambda_6_layer_call_fn_9449922S7¢4
-¢*
 
inputs’’’’’’’’’2

 
p
Ŗ "’’’’’’’’’2°
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449770h7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "-¢*
# 
0’’’’’’’’’
 °
D__inference_re_lu_2_layer_call_and_return_conditional_losses_9449901h7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "-¢*
# 
0’’’’’’’’’
 
)__inference_re_lu_2_layer_call_fn_9449765[7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ " ’’’’’’’’’
)__inference_re_lu_2_layer_call_fn_9449896[7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ " ’’’’’’’’’Ā
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448925uG¢D
=¢:
0-
lambda_4_input’’’’’’’’’
p 

 
Ŗ "&¢#

0’’’’’’’’’
 Ā
I__inference_sequential_4_layer_call_and_return_conditional_losses_9448937uG¢D
=¢:
0-
lambda_4_input’’’’’’’’’
p

 
Ŗ "&¢#

0’’’’’’’’’
 Ā
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449276uG¢D
=¢:
0-
lambda_4_input’’’’’’’’’
p 

 
Ŗ "&¢#

0’’’’’’’’’
 Ā
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449288uG¢D
=¢:
0-
lambda_4_input’’’’’’’’’
p

 
Ŗ "&¢#

0’’’’’’’’’
 ŗ
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449515m?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "&¢#

0’’’’’’’’’
 ŗ
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449531m?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "&¢#

0’’’’’’’’’
 ŗ
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449633m?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "&¢#

0’’’’’’’’’
 ŗ
I__inference_sequential_4_layer_call_and_return_conditional_losses_9449649m?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "&¢#

0’’’’’’’’’
 
.__inference_sequential_4_layer_call_fn_9448834hG¢D
=¢:
0-
lambda_4_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_9448913hG¢D
=¢:
0-
lambda_4_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_9449185hG¢D
=¢:
0-
lambda_4_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_9449264hG¢D
=¢:
0-
lambda_4_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_9449490`?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_9449499`?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_9449608`?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
.__inference_sequential_4_layer_call_fn_9449617`?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’¹
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449120l?¢<
5¢2
(%
lambda_5_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’

 ¹
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449130l?¢<
5¢2
(%
lambda_5_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’

 ¹
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449471l?¢<
5¢2
(%
lambda_5_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’

 ¹
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449481l?¢<
5¢2
(%
lambda_5_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’

 ±
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449583d7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’

 ±
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449599d7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’

 ±
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449701d7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’

 ±
I__inference_sequential_5_layer_call_and_return_conditional_losses_9449717d7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’

 
.__inference_sequential_5_layer_call_fn_9449042_?¢<
5¢2
(%
lambda_5_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’

.__inference_sequential_5_layer_call_fn_9449110_?¢<
5¢2
(%
lambda_5_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

.__inference_sequential_5_layer_call_fn_9449393_?¢<
5¢2
(%
lambda_5_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’

.__inference_sequential_5_layer_call_fn_9449461_?¢<
5¢2
(%
lambda_5_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

.__inference_sequential_5_layer_call_fn_9449558W7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’

.__inference_sequential_5_layer_call_fn_9449567W7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

.__inference_sequential_5_layer_call_fn_9449676W7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’

.__inference_sequential_5_layer_call_fn_9449685W7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
µ
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448994h?¢<
5¢2
(%
lambda_6_input’’’’’’’’’2
p 

 
Ŗ "%¢"

0’’’’’’’’’2
 µ
I__inference_sequential_6_layer_call_and_return_conditional_losses_9448999h?¢<
5¢2
(%
lambda_6_input’’’’’’’’’2
p

 
Ŗ "%¢"

0’’’’’’’’’2
 µ
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449345h?¢<
5¢2
(%
lambda_6_input’’’’’’’’’2
p 

 
Ŗ "%¢"

0’’’’’’’’’2
 µ
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449350h?¢<
5¢2
(%
lambda_6_input’’’’’’’’’2
p

 
Ŗ "%¢"

0’’’’’’’’’2
 ­
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449545`7¢4
-¢*
 
inputs’’’’’’’’’2
p 

 
Ŗ "%¢"

0’’’’’’’’’2
 ­
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449549`7¢4
-¢*
 
inputs’’’’’’’’’2
p

 
Ŗ "%¢"

0’’’’’’’’’2
 ­
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449663`7¢4
-¢*
 
inputs’’’’’’’’’2
p 

 
Ŗ "%¢"

0’’’’’’’’’2
 ­
I__inference_sequential_6_layer_call_and_return_conditional_losses_9449667`7¢4
-¢*
 
inputs’’’’’’’’’2
p

 
Ŗ "%¢"

0’’’’’’’’’2
 
.__inference_sequential_6_layer_call_fn_9448954[?¢<
5¢2
(%
lambda_6_input’’’’’’’’’2
p 

 
Ŗ "’’’’’’’’’2
.__inference_sequential_6_layer_call_fn_9448989[?¢<
5¢2
(%
lambda_6_input’’’’’’’’’2
p

 
Ŗ "’’’’’’’’’2
.__inference_sequential_6_layer_call_fn_9449305[?¢<
5¢2
(%
lambda_6_input’’’’’’’’’2
p 

 
Ŗ "’’’’’’’’’2
.__inference_sequential_6_layer_call_fn_9449340[?¢<
5¢2
(%
lambda_6_input’’’’’’’’’2
p

 
Ŗ "’’’’’’’’’2
.__inference_sequential_6_layer_call_fn_9449536S7¢4
-¢*
 
inputs’’’’’’’’’2
p 

 
Ŗ "’’’’’’’’’2
.__inference_sequential_6_layer_call_fn_9449541S7¢4
-¢*
 
inputs’’’’’’’’’2
p

 
Ŗ "’’’’’’’’’2
.__inference_sequential_6_layer_call_fn_9449654S7¢4
-¢*
 
inputs’’’’’’’’’2
p 

 
Ŗ "’’’’’’’’’2
.__inference_sequential_6_layer_call_fn_9449659S7¢4
-¢*
 
inputs’’’’’’’’’2
p

 
Ŗ "’’’’’’’’’2½
%__inference_signature_wrapper_9448758¤¢ 
¢ 
Ŗ
.

0/discount 

0/discount’’’’’’’’’
L
0/observation/image52
0/observation/image’’’’’’’’’
J
0/observation/random_z0-
0/observation/random_z’’’’’’’’’2
L
0/observation/time_step1.
0/observation/time_step’’’’’’’’’
*
0/reward
0/reward’’’’’’’’’
0
0/step_type!
0/step_type’’’’’’’’’
M
1/actor_network_state/02/
1/actor_network_state/0’’’’’’’’’
M
1/actor_network_state/12/
1/actor_network_state/1’’’’’’’’’"ŚŖÖ
&
action
action’’’’’’’’’	
U
state/actor_network_state/063
state/actor_network_state/0’’’’’’’’’
U
state/actor_network_state/163
state/actor_network_state/1’’’’’’’’’ł
%__inference_signature_wrapper_9448767Ļ0¢-
¢ 
&Ŗ#
!

batch_size

batch_size "Ŗ
I
actor_network_state/00-
actor_network_state/0’’’’’’’’’
I
actor_network_state/10-
actor_network_state/1’’’’’’’’’Y
%__inference_signature_wrapper_94487750¢

¢ 
Ŗ "Ŗ

int64
int64 	=
%__inference_signature_wrapper_9448779¢

¢ 
Ŗ "Ŗ 