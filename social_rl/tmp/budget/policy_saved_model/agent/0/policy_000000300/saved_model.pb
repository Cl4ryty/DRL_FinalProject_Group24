??
?"?"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
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
validate_shapebool( ?
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
?
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
incompatible_shape_errorbool(?
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
delete_old_dirsbool(?
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
2	?

NoOp
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
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
?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??
?
"agent/ValueRnnNetwork/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"agent/ValueRnnNetwork/dense_5/bias
?
6agent/ValueRnnNetwork/dense_5/bias/Read/ReadVariableOpReadVariableOp"agent/ValueRnnNetwork/dense_5/bias*
_output_shapes
:*
dtype0
?
$agent/ValueRnnNetwork/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*5
shared_name&$agent/ValueRnnNetwork/dense_5/kernel
?
8agent/ValueRnnNetwork/dense_5/kernel/Read/ReadVariableOpReadVariableOp$agent/ValueRnnNetwork/dense_5/kernel*
_output_shapes

:(*
dtype0
?
;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias
?
Oagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias/Read/ReadVariableOpReadVariableOp;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias*
_output_shapes	
:?*
dtype0
?
Gagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*X
shared_nameIGagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel
?
[agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpGagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel*
_output_shapes
:	(?*
dtype0
?
=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*N
shared_name?=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel
?
Qagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel/Read/ReadVariableOpReadVariableOp=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel*
_output_shapes
:	 ?*
dtype0
?
Bagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias
?
Vagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias/Read/ReadVariableOpReadVariableOpBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias*
_output_shapes
: *
dtype0
?
Dagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *U
shared_nameFDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel
?
Xagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel/Read/ReadVariableOpReadVariableOpDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel*
_output_shapes

:  *
dtype0
?
Bagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias
?
Vagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOpBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias*
_output_shapes
: *
dtype0
?
Dagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:M *U
shared_nameFDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel
?
Xagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOpDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel*
_output_shapes

:M *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
?
Jagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
?
^agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpReadVariableOpJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
_output_shapes
:*
dtype0
?
Lagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*]
shared_nameNLagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
?
`agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpReadVariableOpLagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel*
_output_shapes
:	?*
dtype0
?
Qagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*b
shared_nameSQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias
?
eagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOpReadVariableOpQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias*
_output_shapes	
:?*
dtype0
?
]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*n
shared_name_]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel
?
qagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpReadVariableOp]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
Sagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*d
shared_nameUSagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel
?
gagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpReadVariableOpSagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel*
_output_shapes
:	 ?*
dtype0
?
Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *k
shared_name\Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias
?
nagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOpZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias*
_output_shapes
: *
dtype0
?
\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *m
shared_name^\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel
?
pagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel*
_output_shapes

:  *
dtype0
?
Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *k
shared_name\Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias
?
nagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias*
_output_shapes
: *
dtype0
?
\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:M *m
shared_name^\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel
?
pagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel*
_output_shapes

:M *
dtype0
r
conv2d/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias_1
k
!conv2d/bias_1/Read/ReadVariableOpReadVariableOpconv2d/bias_1*
_output_shapes
:*
dtype0
?
conv2d/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel_1
{
#conv2d/kernel_1/Read/ReadVariableOpReadVariableOpconv2d/kernel_1*&
_output_shapes
:*
dtype0
p
dense/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias_1
i
 dense/bias_1/Read/ReadVariableOpReadVariableOpdense/bias_1*
_output_shapes
:*
dtype0
x
dense/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel_1
q
"dense/kernel_1/Read/ReadVariableOpReadVariableOpdense/kernel_1*
_output_shapes

:*
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
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
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
?
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
TN
VARIABLE_VALUEdense/kernel_1,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense/bias_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv2d/kernel_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv2d/bias_1,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUESagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUELagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense/kernel-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUE
dense/bias-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv2d/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEconv2d/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUE=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUE;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias-model_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE$agent/ValueRnnNetwork/dense_5/kernel-model_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE"agent/ValueRnnNetwork/dense_5/bias-model_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
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
?
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
?
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
?
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
?
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
?
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
?
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
?
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
?
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
?
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
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_input_tensor_spec
?_preprocessing_nest
?_flat_preprocessing_layers
?_preprocessing_combiner
?_postprocessing_layers*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?cell*
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_input_tensor_spec
?_preprocessing_nest
?_flat_preprocessing_layers
?_preprocessing_combiner
?_postprocessing_layers*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?cell*
* 

'0
(1*

'0
(1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 

?0
?1*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?0
?1
?2*

0
1
2*

0
1
2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
?
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 

?0
?1*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?0
?1
?2*

$0
%1
&2*

$0
%1
&2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
?
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
4
?0
?1
?2
?3
?4
?5*
* 
* 
* 
?
?layer-0
?layer_with_weights-0
?layer-1
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?layer-0
?layer_with_weights-0
?layer-1
?layer-2
?layer-3
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
* 

?0*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
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
4
?0
?1
?2
?3
?4
?5*
* 
* 
* 
?
?layer-0
?layer_with_weights-0
?layer-1
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?layer-0
?layer_with_weights-0
?layer-1
?layer-2
?layer-3
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

 kernel
!bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

"kernel
#bias*
* 

?0*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

 0
!1*

 0
!1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

"0
#1*

"0
#1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

?0
?1*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
$
?0
?1
?2
?3*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

?0
?1*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
$
?0
?1
?2
?3*
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
:?????????*
dtype0*
shape:?????????
?
action_0_observation_directionPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
action_0_observation_imagePlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
j
action_0_rewardPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
action_0_step_typePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
action_1_actor_network_state_0Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
action_1_actor_network_state_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?

StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_directionaction_0_observation_imageaction_0_rewardaction_0_step_typeaction_1_actor_network_state_0action_1_actor_network_state_1dense/kernel_1dense/bias_1conv2d/kernel_1conv2d/bias_1\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernelZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/biasSagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasLagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:?????????:??????????:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_9445707
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_9445716
?
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
GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_9445728
?
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
GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_9445724
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameglobal_step/Read/ReadVariableOp"dense/kernel_1/Read/ReadVariableOp dense/bias_1/Read/ReadVariableOp#conv2d/kernel_1/Read/ReadVariableOp!conv2d/bias_1/Read/ReadVariableOppagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpnagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOppagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpnagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpgagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpqagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpeagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOp`agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOp^agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOpXagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpVagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpXagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel/Read/ReadVariableOpVagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias/Read/ReadVariableOpQagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel/Read/ReadVariableOp[agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel/Read/ReadVariableOpOagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias/Read/ReadVariableOp8agent/ValueRnnNetwork/dense_5/kernel/Read/ReadVariableOp6agent/ValueRnnNetwork/dense_5/bias/Read/ReadVariableOpConst*(
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_9446848
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameglobal_stepdense/kernel_1dense/bias_1conv2d/kernel_1conv2d/bias_1\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernelZagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/biasSagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelQagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasLagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelJagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasdense/kernel
dense/biasconv2d/kernelconv2d/biasDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernelBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/biasDagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernelBagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernelGagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias$agent/ValueRnnNetwork/dense_5/kernel"agent/ValueRnnNetwork/dense_5/bias*'
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_9446939??
?
^
B__inference_re_lu_layer_call_and_return_conditional_losses_9446721

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_9445771
lambda_1_input
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445764o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_1_input
?x
?
#__inference__traced_restore_9446939
file_prefix&
assignvariableop_global_step:	 3
!assignvariableop_1_dense_kernel_1:-
assignvariableop_2_dense_bias_1:<
"assignvariableop_3_conv2d_kernel_1:.
 assignvariableop_4_conv2d_bias_1:?
oassignvariableop_5_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel:M {
massignvariableop_6_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias: ?
oassignvariableop_7_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_kernel:  {
massignvariableop_8_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_bias: y
fassignvariableop_9_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel:	 ??
qassignvariableop_10_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel:
??t
eassignvariableop_11_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias:	?s
`assignvariableop_12_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel:	?l
^assignvariableop_13_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias:2
 assignvariableop_14_dense_kernel:,
assignvariableop_15_dense_bias:;
!assignvariableop_16_conv2d_kernel:-
assignvariableop_17_conv2d_bias:j
Xassignvariableop_18_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel:M d
Vassignvariableop_19_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias: j
Xassignvariableop_20_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_kernel:  d
Vassignvariableop_21_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_bias: d
Qassignvariableop_22_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel:	 ?n
[assignvariableop_23_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel:	(?^
Oassignvariableop_24_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias:	?J
8assignvariableop_25_agent_valuernnnetwork_dense_5_kernel:(D
6assignvariableop_26_agent_valuernnnetwork_dense_5_bias:
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_global_stepIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_kernel_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_bias_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpoassignvariableop_5_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpmassignvariableop_6_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpoassignvariableop_7_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpmassignvariableop_8_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpfassignvariableop_9_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpqassignvariableop_10_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpeassignvariableop_11_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp`assignvariableop_12_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp^assignvariableop_13_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv2d_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_conv2d_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpXassignvariableop_18_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpVassignvariableop_19_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpXassignvariableop_20_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpVassignvariableop_21_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpQassignvariableop_22_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp[assignvariableop_23_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpOassignvariableop_24_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp8assignvariableop_25_agent_valuernnnetwork_dense_5_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp6assignvariableop_26_agent_valuernnnetwork_dense_5_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: ?
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
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445859
lambda_1_input
dense_9445853:
dense_9445855:
identity??dense/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445799?
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9445853dense_9445855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445757u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_1_input
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446240

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference_action_1944565
time_step_step_type
time_step_reward
time_step_discount#
time_step_observation_direction
time_step_observation_image&
"policy_state_actor_network_state_0&
"policy_state_actor_network_state_1?
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource:?
zactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource:?
xactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource:?
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource:?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource: ?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource: ?
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:	 ??
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:
??
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	?q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	?m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2??fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp?oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp?qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp?pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp?gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp?hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpG
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
valueB:?
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
valueB:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????I
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : c
EqualEqualtime_step_step_typeEqual/y:output:0*
T0*#
_output_shapes
:?????????F
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
?????????b
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
:??????????
SelectV2SelectV2Reshape:output:0zeros:output:0"policy_state_actor_network_state_0*
T0*(
_output_shapes
:???????????

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0"policy_state_actor_network_state_1*
T0*(
_output_shapes
:??????????I
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
valueB:?
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
valueB:?O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????K
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : g
Equal_1Equaltime_step_step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:?????????H
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
?????????h
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
:??????????

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:???????????

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:???????????
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_observation_directionOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_observation_imageQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*3
_output_shapes!
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimstime_step_step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:??????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:??????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpzactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAddBiasAddkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul:product:0yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truedivRealDivbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/Cast:y:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/y:output:0*
T0*/
_output_shapes
:??????????
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpxactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2DConv2DeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv:z:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAddBiasAddiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/ReluRelujActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ReshapeReshapekActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/Relu:activations:0iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:?????????H?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concatConcatV2kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????M?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M   ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????M?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:????????? ?
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:??????????
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:????????? ?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:???????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
??????????
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????T
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
valueB ?
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
valueB:?
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
valueB ?
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
value	B : ?
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:?????????u
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
valueB:?
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
value	B : ?
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:?????????Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:?????????Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:?????????\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:??????????

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpp^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpr^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp2?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp2?
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp2?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2?
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2?
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:?????????
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:?????????
,
_user_specified_nametime_step/discount:hd
'
_output_shapes
:?????????
9
_user_specified_name!time_step/observation/direction:lh
/
_output_shapes
:?????????
5
_user_specified_nametime_step/observation/image:lh
(
_output_shapes
:??????????
<
_user_specified_name$"policy_state/actor_network_state/0:lh
(
_output_shapes
:??????????
<
_user_specified_name$"policy_state/actor_network_state/1
?
?
.__inference_sequential_1_layer_call_fn_9446324

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445823o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_function_with_signature_1943997
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:M 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 ?
	unknown_8:
??
	unknown_9:	?

unknown_10:	?

unknown_11:
identity	

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_directionobservation_imageactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:?????????:??????????:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_action_1943964k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:?????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:OK
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:`\
'
_output_shapes
:?????????
1
_user_specified_name0/observation/direction:d`
/
_output_shapes
:?????????
-
_user_specified_name0/observation/image:a]
(
_output_shapes
:??????????
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:??????????
1
_user_specified_name1/actor_network_state/1
?
?
.__inference_sequential_1_layer_call_fn_9446315

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445764o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446490

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOpd
lambda/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????U
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda/truedivRealDivlambda/Cast:y:0lambda/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dlambda/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????e

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ~
flatten/ReshapeReshapere_lu/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????Hg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446692

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446294
lambda_input(
conv2d_9446286:
conv2d_9446288:
identity??conv2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446162?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446286conv2d_9446288*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446174?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446185?
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446193o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_namelambda_input
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446017
lambda_input(
conv2d_9446009:
conv2d_9446011:
identity??conv2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445951?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446009conv2d_9446011*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445885?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445896?
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445904o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_namelambda_input
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446196

inputs(
conv2d_9446175:
conv2d_9446177:
identity??conv2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446162?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446175conv2d_9446177*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446174?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446185?
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446193o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446162

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_9446203
lambda_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_namelambda_input
?
F
*__inference_lambda_1_layer_call_fn_9446511

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445745`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_9446668

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9445907

inputs(
conv2d_9445886:
conv2d_9445888:
identity??conv2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445873?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9445886conv2d_9445888*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445885?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445896?
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445904o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9446619

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????HX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_9446046

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
-
+__inference_function_with_signature_1944091?
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
GPU 2J 8? *!
fR
__inference_<lambda>_922*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446356

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
lambda_1/one_hotOneHotinputslambda_1/one_hot/depth:output:0"lambda_1/one_hot/on_value:output:0#lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????g
lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lambda_1/ReshapeReshapelambda_1/one_hot:output:0lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMullambda_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?H
?
 __inference__traced_save_9446848
file_prefix*
&savev2_global_step_read_readvariableop	-
)savev2_dense_kernel_1_read_readvariableop+
'savev2_dense_bias_1_read_readvariableop.
*savev2_conv2d_kernel_1_read_readvariableop,
(savev2_conv2d_bias_1_read_readvariableop{
wsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableopy
usavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableop{
wsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_kernel_read_readvariableopy
usavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_bias_read_readvariableopr
nsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel_read_readvariableop|
xsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableopp
lsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias_read_readvariableopk
gsavev2_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopi
esavev2_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableopc
_savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel_read_readvariableopa
]savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias_read_readvariableopc
_savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_kernel_read_readvariableopa
]savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_bias_read_readvariableop\
Xsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel_read_readvariableopf
bsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel_read_readvariableopZ
Vsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias_read_readvariableopC
?savev2_agent_valuernnnetwork_dense_5_kernel_read_readvariableopA
=savev2_agent_valuernnnetwork_dense_5_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_global_step_read_readvariableop)savev2_dense_kernel_1_read_readvariableop'savev2_dense_bias_1_read_readvariableop*savev2_conv2d_kernel_1_read_readvariableop(savev2_conv2d_bias_1_read_readvariableopwsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableopusavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableopwsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_kernel_read_readvariableopusavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_bias_read_readvariableopnsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel_read_readvariableopxsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableoplsavev2_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias_read_readvariableopgsavev2_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopesavev2_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop_savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel_read_readvariableop]savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias_read_readvariableop_savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_kernel_read_readvariableop]savev2_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_4_bias_read_readvariableopXsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_kernel_read_readvariableopbsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_recurrent_kernel_read_readvariableopVsavev2_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_bias_read_readvariableop?savev2_agent_valuernnnetwork_dense_5_kernel_read_readvariableop=savev2_agent_valuernnnetwork_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : :::::M : :  : :	 ?:
??:?:	?::::::M : :  : :	 ?:	(?:?:(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:M : 
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
:	 ?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:M : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :%!

_output_shapes
:	 ?:%!

_output_shapes
:	(?:!

_output_shapes	
:?:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: 
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_9446555

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445849
lambda_1_input
dense_9445843:
dense_9445845:
identity??dense/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445745?
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9445843dense_9445845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445757u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_1_input
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446579

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
D
(__inference_lambda_layer_call_fn_9446673

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446162h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_9446060
lambda_1_input
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_1_input
?
^
B__inference_re_lu_layer_call_and_return_conditional_losses_9446608

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445823

inputs
dense_9445817:
dense_9445819:
identity??dense/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445799?
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9445817dense_9445819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445757u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9446732

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????HX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_layer_call_and_return_conditional_losses_9445885

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_layer_call_fn_9446726

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446193`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
%__inference_signature_wrapper_9445724
unknown:	 
identity	??StatefulPartitionedCall?
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
GPU 2J 8? *4
f/R-
+__inference_function_with_signature_1944080^
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
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446456

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
lambda_1/one_hotOneHotinputslambda_1/one_hot/depth:output:0"lambda_1/one_hot/on_value:output:0#lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????g
lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lambda_1/ReshapeReshapelambda_1/one_hot:output:0lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMullambda_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_layer_call_fn_9446658

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446046o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_9445914
lambda_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9445907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_namelambda_input
?
D
(__inference_lambda_layer_call_fn_9446678

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446240h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446711

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ȓ
?
__inference_action_1943964
	time_step
time_step_1
time_step_2
time_step_3
time_step_4
policy_state
policy_state_1?
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource:?
zactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource:?
xactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource:?
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource:?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource: ?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource: ?
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:	 ??
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:
??
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	?q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	?m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2??fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp?oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp?qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp?pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp?gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp?hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp@
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
valueB:?
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
valueB:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????I
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	time_stepEqual/y:output:0*
T0*#
_output_shapes
:?????????F
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
?????????b
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
:?????????w
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state*
T0*(
_output_shapes
:??????????}

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*(
_output_shapes
:??????????B
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
valueB:?
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
valueB:?O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????K
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	time_stepEqual_1/y:output:0*
T0*#
_output_shapes
:?????????H
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
?????????h
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
:??????????

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:???????????

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:???????????
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_3OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_4QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*3
_output_shapes!
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDims	time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:??????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:??????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpzactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAddBiasAddkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul:product:0yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truedivRealDivbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/Cast:y:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/y:output:0*
T0*/
_output_shapes
:??????????
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpxactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2DConv2DeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv:z:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAddBiasAddiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/ReluRelujActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ReshapeReshapekActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/Relu:activations:0iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:?????????H?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concatConcatV2kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????M?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M   ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????M?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:????????? ?
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:??????????
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:????????? ?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:???????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
??????????
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????T
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
valueB ?
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
valueB:?
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
valueB ?
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
value	B : ?
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:?????????u
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
valueB:?
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
value	B : ?
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:?????????Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:?????????Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:?????????\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:??????????

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpp^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpr^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp2?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp2?
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp2?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2?
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2?
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:RN
'
_output_shapes
:?????????
#
_user_specified_name	time_step:ZV
/
_output_shapes
:?????????
#
_user_specified_name	time_step:VR
(
_output_shapes
:??????????
&
_user_specified_namepolicy_state:VR
(
_output_shapes
:??????????
&
_user_specified_namepolicy_state
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446005
lambda_input(
conv2d_9445997:
conv2d_9445999:
identity??conv2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445873?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9445997conv2d_9445999*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445885?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445896?
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445904o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_namelambda_input
?
U
%__inference_signature_wrapper_9445716

batch_size
identity

identity_1?
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_function_with_signature_1944064a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????c

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:??????????"
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
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9445977

inputs(
conv2d_9445969:
conv2d_9445971:
identity??conv2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445951?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9445969conv2d_9445971*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445885?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445896?
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445904o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_9446415

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446639

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
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
value	B :?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446598

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445745

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
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
value	B :?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_9445707
discount
observation_direction
observation_image

reward
	step_type
actor_network_state_0
actor_network_state_1
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:M 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 ?
	unknown_8:
??
	unknown_9:	?

unknown_10:	?

unknown_11:
identity	

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_directionobservation_imageactor_network_state_0actor_network_state_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:?????????:??????????:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_function_with_signature_1943997k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:?????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:`\
'
_output_shapes
:?????????
1
_user_specified_name0/observation/direction:d`
/
_output_shapes
:?????????
-
_user_specified_name0/observation/image:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:a]
(
_output_shapes
:??????????
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:??????????
1
_user_specified_name1/actor_network_state/1
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446526

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
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
value	B :?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446685

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_9446424

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446112o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
__inference_<lambda>_919!
readvariableop_resource:	 
identity	??ReadVariableOp^
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
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_9446572

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9445904

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????HX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
k
+__inference_function_with_signature_1944080
unknown:	 
identity	??StatefulPartitionedCall?
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
GPU 2J 8? *!
fR
__inference_<lambda>_919^
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
?

?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446112

inputs
dense_9446106:
dense_9446108:
identity??dense/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446088?
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9446106dense_9446108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446046u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_re_lu_layer_call_fn_9446603

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9445896h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_lambda_1_layer_call_fn_9446516

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445799`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
U
%__inference_get_initial_state_1944059

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
valueB:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????W
IdentityIdentityzeros:output:0*
T0*(
_output_shapes
:??????????[

Identity_1Identityzeros_1:output:0*
T0*(
_output_shapes
:??????????"
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
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446340

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
lambda_1/one_hotOneHotinputslambda_1/one_hot/depth:output:0"lambda_1/one_hot/on_value:output:0#lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????g
lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lambda_1/ReshapeReshapelambda_1/one_hot:output:0lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMullambda_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_re_lu_layer_call_fn_9446716

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446185h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446148
lambda_1_input
dense_9446142:
dense_9446144:
identity??dense/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446088?
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9446142dense_9446144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446046u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_1_input
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_9445873

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446034

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
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
value	B :?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
[
+__inference_function_with_signature_1944064

batch_size
identity

identity_1?
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_get_initial_state_1944059a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????c

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:??????????"
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
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446649

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
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
value	B :?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_layer_call_fn_9446613

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9445904`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_lambda_1_layer_call_fn_9446629

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446088`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445799

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
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
value	B :?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445764

inputs
dense_9445758:
dense_9445760:
identity??dense/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9445745?
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9445758dense_9445760*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445757u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446138
lambda_1_input
dense_9446132:
dense_9446134:
identity??dense/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCalllambda_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446034?
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9446132dense_9446134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446046u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_1_input
?
?
,__inference_sequential_layer_call_fn_9446374

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9445977o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
'
%__inference_signature_wrapper_9445728?
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
GPU 2J 8? *4
f/R-
+__inference_function_with_signature_1944091*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446440

inputs6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp^
lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
lambda_1/one_hotOneHotinputslambda_1/one_hot/depth:output:0"lambda_1/one_hot/on_value:output:0#lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????g
lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lambda_1/ReshapeReshapelambda_1/one_hot:output:0lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMullambda_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
U
%__inference_get_initial_state_1944791

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
valueB:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????W
IdentityIdentityzeros:output:0*
T0*(
_output_shapes
:??????????[

Identity_1Identityzeros_1:output:0*
T0*(
_output_shapes
:??????????"
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
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_9445757

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446174

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_lambda_1_layer_call_fn_9446624

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446034`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_layer_call_and_return_conditional_losses_9445896

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_lambda_layer_call_and_return_conditional_losses_9445951

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446088

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
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
value	B :?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_9446474

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446266o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_9446282
lambda_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446266o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_namelambda_input
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446266

inputs(
conv2d_9446258:
conv2d_9446260:
identity??conv2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446240?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446258conv2d_9446260*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446174?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446185?
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446193o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Y

__inference_<lambda>_922*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
__inference_action_1944330
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1?
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource:?
zactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource:?
xactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource:?
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource:?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource: ?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource: ?
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:	 ??
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:
??
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	?q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	?m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2??fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp?oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp?qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp?pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp?gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp?hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
valueB:?
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
valueB:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????I
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:?????????F
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
?????????b
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
:??????????
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0*
T0*(
_output_shapes
:???????????

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_1*
T0*(
_output_shapes
:???????????
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
valueB:?
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
valueB:?O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????K
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:?????????H
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
?????????h
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
:??????????

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:???????????

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:???????????
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservation_directionOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_imageQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*3
_output_shapes!
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:??????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:??????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpzactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAddBiasAddkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul:product:0yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truedivRealDivbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/Cast:y:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/y:output:0*
T0*/
_output_shapes
:??????????
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpxactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2DConv2DeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv:z:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAddBiasAddiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/ReluRelujActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ReshapeReshapekActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/Relu:activations:0iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:?????????H?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concatConcatV2kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????M?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M   ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????M?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:????????? ?
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:??????????
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:????????? ?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:???????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
??????????
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????T
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
valueB ?
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
valueB:?
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
valueB ?
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
value	B : ?
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 Deterministic/sample/BroadcastToBroadcastTohCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:?????????u
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
valueB:?
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
value	B : ?
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:?????????Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:?????????Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:?????????\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:??????????

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpp^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpr^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp2?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp2?
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp2?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2?
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2?
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:^Z
'
_output_shapes
:?????????
/
_user_specified_nameobservation/direction:b^
/
_output_shapes
:?????????
+
_user_specified_nameobservation/image:_[
(
_output_shapes
:??????????
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:??????????
/
_user_specified_nameactor_network_state/1
?
D
(__inference_lambda_layer_call_fn_9446560

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445873h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_9446128
lambda_1_input
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446112o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_1_input
?
D
(__inference_lambda_layer_call_fn_9446565

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9445951h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_9446465

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9446196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_9445839
lambda_1_input
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445823o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_1_input
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446390

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOpd
lambda/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????U
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda/truedivRealDivlambda/Cast:y:0lambda/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dlambda/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????e

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ~
flatten/ReshapeReshapere_lu/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????Hg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_layer_call_fn_9446588

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9445885w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446406

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOpd
lambda/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????U
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda/truedivRealDivlambda/Cast:y:0lambda/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dlambda/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????e

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ~
flatten/ReshapeReshapere_lu/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????Hg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_layer_call_fn_9446701

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446174w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_9445993
lambda_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9445977o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_namelambda_input
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446306
lambda_input(
conv2d_9446298:
conv2d_9446300:
identity??conv2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_9446240?
conv2d/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0conv2d_9446298conv2d_9446300*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9446174?
re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_9446185?
flatten/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9446193o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_namelambda_input
?
a
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446536

inputs
identityU
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
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
value	B :?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_layer_call_fn_9446545

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9445757o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9446193

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????HX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????H"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
#__inference_distribution_fn_1944775
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1?
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource:?
zactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource:?
xactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource:?
yactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource:?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource: ?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource: ?
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:	 ??
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:
??
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	?q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	?m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1	

identity_2	

identity_3

identity_4??fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp?oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp?qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp?pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp?gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp?hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
valueB:?
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
valueB:?M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????I
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:?????????F
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
?????????b
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
:??????????
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0*
T0*(
_output_shapes
:???????????

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_1*
T0*(
_output_shapes
:???????????
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
valueB:?
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
valueB:?O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????\
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:?O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
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
:??????????K
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:?????????H
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
?????????h
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
:??????????

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*(
_output_shapes
:???????????

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:???????????
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservation_directionOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_imageQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*3
_output_shapes!
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:??????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:??????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/lambda_1/Reshape:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpzactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAddBiasAddkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul:product:0yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truedivRealDivbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/Cast:y:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv/y:output:0*
T0*/
_output_shapes
:??????????
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpxactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2DConv2DeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/lambda/truediv:z:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpyactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAddBiasAddiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/ReluRelujActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/ReshapeReshapekActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/re_lu/Relu:activations:0iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:?????????H?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concatConcatV2kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/flatten/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????M?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M   ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????M?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	?
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:????????? ?
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:??????????
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:????????? ?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*(
_output_shapes
:???????????
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:???????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
iCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
??????????
_Categorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMaxArgMaxIActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape:output:0rCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????T
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
: ?

Identity_1IdentityhCategorical_CONSTRUCTED_AT_ActorDistributionRnnNetwork_CategoricalProjectionNetwork/mode/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:?????????[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0	*
_output_shapes
: ?

Identity_3IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_4IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOpp^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpr^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpq^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/BiasAdd/ReadVariableOp2?
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOpoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential/conv2d/Conv2D/ReadVariableOp2?
qActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOpqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/BiasAdd/ReadVariableOp2?
pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOppActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_1/dense/MatMul/ReadVariableOp2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2?
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2?
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpUActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:^Z
'
_output_shapes
:?????????
/
_user_specified_nameobservation/direction:b^
/
_output_shapes
:?????????
+
_user_specified_nameobservation/image:_[
(
_output_shapes
:??????????
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:??????????
/
_user_specified_nameactor_network_state/1
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_9446506

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOpd
lambda/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????U
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
lambda/truedivRealDivlambda/Cast:y:0lambda/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dlambda/truediv:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????e

re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ~
flatten/ReshapeReshapere_lu/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????Hg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446053

inputs
dense_9446047:
dense_9446049:
identity??dense/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446034?
dense/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_9446047dense_9446049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9446046u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_9446365

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_9445907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_layer_call_and_return_conditional_losses_9446185

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0_discount:0?????????
R
0/observation/direction7
 action_0_observation_direction:0?????????
R
0/observation/image;
action_0_observation_image:0?????????
0
0/reward$
action_0_reward:0?????????
6
0/step_type'
action_0_step_type:0?????????
S
1/actor_network_state/08
 action_1_actor_network_state_0:0??????????
S
1/actor_network_state/18
 action_1_actor_network_state_1:0??????????6
action,
StatefulPartitionedCall:0	?????????P
state/actor_network_state/01
StatefulPartitionedCall:1??????????P
state/actor_network_state/11
StatefulPartitionedCall:2??????????tensorflow/serving/predict*?
get_initial_state?
2

batch_size$
get_initial_state_batch_size:0 B
actor_network_state/0)
PartitionedCall:0??????????B
actor_network_state/1)
PartitionedCall:1??????????tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:??
?
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
?
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
?
)_time_step_spec
*_policy_state_spec
+_policy_step_spec
,_trajectory_spec
-_wrapped_policy"
trackable_dict_wrapper
?
.trace_0
/trace_12?
__inference_action_1944330
__inference_action_1944565?
???
FullArgSpec8
args0?-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults?	
? 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z.trace_0z/trace_1
?
0trace_02?
#__inference_distribution_fn_1944775?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z0trace_0
?
1trace_02?
%__inference_get_initial_state_1944791?
???
FullArgSpec!
args?
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z1trace_0
?B?
__inference_<lambda>_922"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_919"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
:2dense/kernel
:2
dense/bias
':%2conv2d/kernel
:2conv2d/bias
n:lM 2\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel
h:f 2Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias
n:l  2\agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/kernel
h:f 2Zagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_2/bias
f:d	 ?2Sagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel
q:o
??2]agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel
`:^?2Qagent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias
_:]	?2Lagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
X:V2Jagent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
:2dense/kernel
:2
dense/bias
':%2conv2d/kernel
:2conv2d/bias
V:TM 2Dagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel
P:N 2Bagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias
V:T  2Dagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/kernel
P:N 2Bagent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_4/bias
P:N	 ?2=agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/kernel
Z:X	(?2Gagent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/recurrent_kernel
J:H?2;agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/bias
6:4(2$agent/ValueRnnNetwork/dense_5/kernel
0:.2"agent/ValueRnnNetwork/dense_5/bias
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
?
6_actor_network
7_time_step_spec
8_policy_state_spec
9_policy_step_spec
:_trajectory_spec
;_value_network"
_generic_user_object
?B?
__inference_action_1944330	step_typerewarddiscountobservation/directionobservation/imageactor_network_state/0actor_network_state/1"?
???
FullArgSpec8
args0?-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults?	
? 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_action_1944565time_step/step_typetime_step/rewardtime_step/discounttime_step/observation/directiontime_step/observation/image"policy_state/actor_network_state/0"policy_state/actor_network_state/1"?
???
FullArgSpec8
args0?-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults?	
? 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_distribution_fn_1944775	step_typerewarddiscountobservation/directionobservation/imageactor_network_state/0actor_network_state/1"?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_get_initial_state_1944791
batch_size"?
???
FullArgSpec!
args?
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_9445707
0/discount0/observation/direction0/observation/image0/reward0/step_type1/actor_network_state/01/actor_network_state/1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_9445716
batch_size"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_9445724"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_9445728"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
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
?
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
?
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
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?	
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?	
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
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
?
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
?
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
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
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
?
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
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?	
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?	
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_input_tensor_spec
?_preprocessing_nest
?_flat_preprocessing_layers
?_preprocessing_combiner
?_postprocessing_layers"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?cell"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec?
args7?4
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
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec?
args7?4
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
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?	
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?	
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_input_tensor_spec
?_preprocessing_nest
?_flat_preprocessing_layers
?_preprocessing_combiner
?_postprocessing_layers"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?cell"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
8
?0
?1
?2"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpecH
args@?=
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
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecH
args@?=
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
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
?
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
8
?0
?1
?2"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpecH
args@?=
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
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecH
args@?=
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
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
?
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
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?layer-0
?layer_with_weights-0
?layer-1
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
?layer-0
?layer_with_weights-0
?layer-1
?layer-2
?layer-3
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
(
?0"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?layer-0
?layer_with_weights-0
?layer-1
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
?layer-0
?layer_with_weights-0
?layer-1
?layer-2
?layer-3
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
 "
trackable_list_wrapper
(
?0"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
.__inference_sequential_1_layer_call_fn_9445771
.__inference_sequential_1_layer_call_fn_9446315
.__inference_sequential_1_layer_call_fn_9446324
.__inference_sequential_1_layer_call_fn_9445839?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446340
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446356
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445849
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445859?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
,__inference_sequential_layer_call_fn_9445914
,__inference_sequential_layer_call_fn_9446365
,__inference_sequential_layer_call_fn_9446374
,__inference_sequential_layer_call_fn_9445993?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
G__inference_sequential_layer_call_and_return_conditional_losses_9446390
G__inference_sequential_layer_call_and_return_conditional_losses_9446406
G__inference_sequential_layer_call_and_return_conditional_losses_9446005
G__inference_sequential_layer_call_and_return_conditional_losses_9446017?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
.__inference_sequential_1_layer_call_fn_9446060
.__inference_sequential_1_layer_call_fn_9446415
.__inference_sequential_1_layer_call_fn_9446424
.__inference_sequential_1_layer_call_fn_9446128?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446440
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446456
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446138
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446148?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
,__inference_sequential_layer_call_fn_9446203
,__inference_sequential_layer_call_fn_9446465
,__inference_sequential_layer_call_fn_9446474
,__inference_sequential_layer_call_fn_9446282?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
G__inference_sequential_layer_call_and_return_conditional_losses_9446490
G__inference_sequential_layer_call_and_return_conditional_losses_9446506
G__inference_sequential_layer_call_and_return_conditional_losses_9446294
G__inference_sequential_layer_call_and_return_conditional_losses_9446306?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_lambda_1_layer_call_fn_9446511
*__inference_lambda_1_layer_call_fn_9446516?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446526
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446536?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_layer_call_fn_9446545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_layer_call_and_return_conditional_losses_9446555?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_sequential_1_layer_call_fn_9445771lambda_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_1_layer_call_fn_9446315inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_1_layer_call_fn_9446324inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_1_layer_call_fn_9445839lambda_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446340inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446356inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445849lambda_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445859lambda_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
(__inference_lambda_layer_call_fn_9446560
(__inference_lambda_layer_call_fn_9446565?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
C__inference_lambda_layer_call_and_return_conditional_losses_9446572
C__inference_lambda_layer_call_and_return_conditional_losses_9446579?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_layer_call_fn_9446588?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446598?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_re_lu_layer_call_fn_9446603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_re_lu_layer_call_and_return_conditional_losses_9446608?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_flatten_layer_call_fn_9446613?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_flatten_layer_call_and_return_conditional_losses_9446619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_sequential_layer_call_fn_9445914lambda_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_layer_call_fn_9446365inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_layer_call_fn_9446374inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_layer_call_fn_9445993lambda_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_layer_call_and_return_conditional_losses_9446390inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_layer_call_and_return_conditional_losses_9446406inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_layer_call_and_return_conditional_losses_9446005lambda_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_layer_call_and_return_conditional_losses_9446017lambda_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_lambda_1_layer_call_fn_9446624
*__inference_lambda_1_layer_call_fn_9446629?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446639
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446649?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_layer_call_fn_9446658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_layer_call_and_return_conditional_losses_9446668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_sequential_1_layer_call_fn_9446060lambda_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_1_layer_call_fn_9446415inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_1_layer_call_fn_9446424inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_1_layer_call_fn_9446128lambda_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446440inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446456inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446138lambda_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446148lambda_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
(__inference_lambda_layer_call_fn_9446673
(__inference_lambda_layer_call_fn_9446678?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
C__inference_lambda_layer_call_and_return_conditional_losses_9446685
C__inference_lambda_layer_call_and_return_conditional_losses_9446692?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_layer_call_fn_9446701?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446711?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_re_lu_layer_call_fn_9446716?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_re_lu_layer_call_and_return_conditional_losses_9446721?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_flatten_layer_call_fn_9446726?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_flatten_layer_call_and_return_conditional_losses_9446732?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_sequential_layer_call_fn_9446203lambda_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_layer_call_fn_9446465inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_layer_call_fn_9446474inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_layer_call_fn_9446282lambda_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_layer_call_and_return_conditional_losses_9446490inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_layer_call_and_return_conditional_losses_9446506inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_layer_call_and_return_conditional_losses_9446294lambda_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_layer_call_and_return_conditional_losses_9446306lambda_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
*__inference_lambda_1_layer_call_fn_9446511inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_lambda_1_layer_call_fn_9446516inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446526inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446536inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
'__inference_dense_layer_call_fn_9446545inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_layer_call_and_return_conditional_losses_9446555inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
(__inference_lambda_layer_call_fn_9446560inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_lambda_layer_call_fn_9446565inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_lambda_layer_call_and_return_conditional_losses_9446572inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_lambda_layer_call_and_return_conditional_losses_9446579inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
(__inference_conv2d_layer_call_fn_9446588inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446598inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
'__inference_re_lu_layer_call_fn_9446603inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_re_lu_layer_call_and_return_conditional_losses_9446608inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
)__inference_flatten_layer_call_fn_9446613inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_flatten_layer_call_and_return_conditional_losses_9446619inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_lambda_1_layer_call_fn_9446624inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_lambda_1_layer_call_fn_9446629inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446639inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446649inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
'__inference_dense_layer_call_fn_9446658inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_layer_call_and_return_conditional_losses_9446668inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
(__inference_lambda_layer_call_fn_9446673inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_lambda_layer_call_fn_9446678inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_lambda_layer_call_and_return_conditional_losses_9446685inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_lambda_layer_call_and_return_conditional_losses_9446692inputs"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
(__inference_conv2d_layer_call_fn_9446701inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446711inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
'__inference_re_lu_layer_call_fn_9446716inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_re_lu_layer_call_and_return_conditional_losses_9446721inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
)__inference_flatten_layer_call_fn_9446726inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_flatten_layer_call_and_return_conditional_losses_9446732inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 7
__inference_<lambda>_919?

? 
? "? 	0
__inference_<lambda>_922?

? 
? "? ?
__inference_action_1944330????
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount??????????
observation?|
<
	direction/?,
observation/direction?????????
<
image3?0
observation/image?????????
???
~
actor_network_stateg?d
0?-
actor_network_state/0??????????
0?-
actor_network_state/1??????????

 
? "???

PolicyStep&
action?
action?????????	?
state???
?
actor_network_states?p
6?3
state/actor_network_state/0??????????
6?3
state/actor_network_state/1??????????
info? ?
__inference_action_1944565????
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount??????????
observation???
F
	direction9?6
time_step/observation/direction?????????
F
image=?:
time_step/observation/image?????????
???
?
actor_network_state??~
=?:
"policy_state/actor_network_state/0??????????
=?:
"policy_state/actor_network_state/1??????????

 
? "???

PolicyStep&
action?
action?????????	?
state???
?
actor_network_states?p
6?3
state/actor_network_state/0??????????
6?3
state/actor_network_state/1??????????
info? ?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446598l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
C__inference_conv2d_layer_call_and_return_conditional_losses_9446711l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_layer_call_fn_9446588_7?4
-?*
(?%
inputs?????????
? " ???????????
(__inference_conv2d_layer_call_fn_9446701_7?4
-?*
(?%
inputs?????????
? " ???????????
B__inference_dense_layer_call_and_return_conditional_losses_9446555\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
B__inference_dense_layer_call_and_return_conditional_losses_9446668\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_layer_call_fn_9446545O/?,
%?"
 ?
inputs?????????
? "??????????z
'__inference_dense_layer_call_fn_9446658O/?,
%?"
 ?
inputs?????????
? "???????????
#__inference_distribution_fn_1944775????
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount??????????
observation?|
<
	direction/?,
observation/direction?????????
<
image3?0
observation/image?????????
???
~
actor_network_stateg?d
0?-
actor_network_state/0??????????
0?-
actor_network_state/1??????????
? "???

PolicyStep?
action??????
`
B??

atol? 	

loc??????????	

rtol? 	
J?G

allow_nan_statsp

namejDeterministic_1

validate_argsp 
?
j
parameters
? 
?
jname+tfp.distributions.Deterministic_ACTTypeSpec ?
state???
?
actor_network_states?p
6?3
state/actor_network_state/0??????????
6?3
state/actor_network_state/1??????????
info? ?
D__inference_flatten_layer_call_and_return_conditional_losses_9446619`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????H
? ?
D__inference_flatten_layer_call_and_return_conditional_losses_9446732`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????H
? ?
)__inference_flatten_layer_call_fn_9446613S7?4
-?*
(?%
inputs?????????
? "??????????H?
)__inference_flatten_layer_call_fn_9446726S7?4
-?*
(?%
inputs?????????
? "??????????H?
%__inference_get_initial_state_1944791?"?
?
?

batch_size 
? "???
~
actor_network_stateg?d
0?-
actor_network_state/0??????????
0?-
actor_network_state/1???????????
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446526`7?4
-?*
 ?
inputs?????????

 
p 
? "%?"
?
0?????????
? ?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446536`7?4
-?*
 ?
inputs?????????

 
p
? "%?"
?
0?????????
? ?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446639`7?4
-?*
 ?
inputs?????????

 
p 
? "%?"
?
0?????????
? ?
E__inference_lambda_1_layer_call_and_return_conditional_losses_9446649`7?4
-?*
 ?
inputs?????????

 
p
? "%?"
?
0?????????
? ?
*__inference_lambda_1_layer_call_fn_9446511S7?4
-?*
 ?
inputs?????????

 
p 
? "???????????
*__inference_lambda_1_layer_call_fn_9446516S7?4
-?*
 ?
inputs?????????

 
p
? "???????????
*__inference_lambda_1_layer_call_fn_9446624S7?4
-?*
 ?
inputs?????????

 
p 
? "???????????
*__inference_lambda_1_layer_call_fn_9446629S7?4
-?*
 ?
inputs?????????

 
p
? "???????????
C__inference_lambda_layer_call_and_return_conditional_losses_9446572p??<
5?2
(?%
inputs?????????

 
p 
? "-?*
#? 
0?????????
? ?
C__inference_lambda_layer_call_and_return_conditional_losses_9446579p??<
5?2
(?%
inputs?????????

 
p
? "-?*
#? 
0?????????
? ?
C__inference_lambda_layer_call_and_return_conditional_losses_9446685p??<
5?2
(?%
inputs?????????

 
p 
? "-?*
#? 
0?????????
? ?
C__inference_lambda_layer_call_and_return_conditional_losses_9446692p??<
5?2
(?%
inputs?????????

 
p
? "-?*
#? 
0?????????
? ?
(__inference_lambda_layer_call_fn_9446560c??<
5?2
(?%
inputs?????????

 
p 
? " ???????????
(__inference_lambda_layer_call_fn_9446565c??<
5?2
(?%
inputs?????????

 
p
? " ???????????
(__inference_lambda_layer_call_fn_9446673c??<
5?2
(?%
inputs?????????

 
p 
? " ???????????
(__inference_lambda_layer_call_fn_9446678c??<
5?2
(?%
inputs?????????

 
p
? " ???????????
B__inference_re_lu_layer_call_and_return_conditional_losses_9446608h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
B__inference_re_lu_layer_call_and_return_conditional_losses_9446721h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_re_lu_layer_call_fn_9446603[7?4
-?*
(?%
inputs?????????
? " ???????????
'__inference_re_lu_layer_call_fn_9446716[7?4
-?*
(?%
inputs?????????
? " ???????????
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445849l??<
5?2
(?%
lambda_1_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9445859l??<
5?2
(?%
lambda_1_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446138l??<
5?2
(?%
lambda_1_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446148l??<
5?2
(?%
lambda_1_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446340d7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446356d7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446440d7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_9446456d7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_1_layer_call_fn_9445771_??<
5?2
(?%
lambda_1_input?????????
p 

 
? "???????????
.__inference_sequential_1_layer_call_fn_9445839_??<
5?2
(?%
lambda_1_input?????????
p

 
? "???????????
.__inference_sequential_1_layer_call_fn_9446060_??<
5?2
(?%
lambda_1_input?????????
p 

 
? "???????????
.__inference_sequential_1_layer_call_fn_9446128_??<
5?2
(?%
lambda_1_input?????????
p

 
? "???????????
.__inference_sequential_1_layer_call_fn_9446315W7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
.__inference_sequential_1_layer_call_fn_9446324W7?4
-?*
 ?
inputs?????????
p

 
? "???????????
.__inference_sequential_1_layer_call_fn_9446415W7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
.__inference_sequential_1_layer_call_fn_9446424W7?4
-?*
 ?
inputs?????????
p

 
? "???????????
G__inference_sequential_layer_call_and_return_conditional_losses_9446005rE?B
;?8
.?+
lambda_input?????????
p 

 
? "%?"
?
0?????????H
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_9446017rE?B
;?8
.?+
lambda_input?????????
p

 
? "%?"
?
0?????????H
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_9446294rE?B
;?8
.?+
lambda_input?????????
p 

 
? "%?"
?
0?????????H
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_9446306rE?B
;?8
.?+
lambda_input?????????
p

 
? "%?"
?
0?????????H
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_9446390l??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????H
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_9446406l??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????H
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_9446490l??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????H
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_9446506l??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????H
? ?
,__inference_sequential_layer_call_fn_9445914eE?B
;?8
.?+
lambda_input?????????
p 

 
? "??????????H?
,__inference_sequential_layer_call_fn_9445993eE?B
;?8
.?+
lambda_input?????????
p

 
? "??????????H?
,__inference_sequential_layer_call_fn_9446203eE?B
;?8
.?+
lambda_input?????????
p 

 
? "??????????H?
,__inference_sequential_layer_call_fn_9446282eE?B
;?8
.?+
lambda_input?????????
p

 
? "??????????H?
,__inference_sequential_layer_call_fn_9446365_??<
5?2
(?%
inputs?????????
p 

 
? "??????????H?
,__inference_sequential_layer_call_fn_9446374_??<
5?2
(?%
inputs?????????
p

 
? "??????????H?
,__inference_sequential_layer_call_fn_9446465_??<
5?2
(?%
inputs?????????
p 

 
? "??????????H?
,__inference_sequential_layer_call_fn_9446474_??<
5?2
(?%
inputs?????????
p

 
? "??????????H?
%__inference_signature_wrapper_9445707????
? 
???
.

0/discount ?

0/discount?????????
L
0/observation/direction1?.
0/observation/direction?????????
L
0/observation/image5?2
0/observation/image?????????
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????
M
1/actor_network_state/02?/
1/actor_network_state/0??????????
M
1/actor_network_state/12?/
1/actor_network_state/1??????????"???
&
action?
action?????????	
U
state/actor_network_state/06?3
state/actor_network_state/0??????????
U
state/actor_network_state/16?3
state/actor_network_state/1???????????
%__inference_signature_wrapper_9445716?0?-
? 
&?#
!

batch_size?

batch_size "???
I
actor_network_state/00?-
actor_network_state/0??????????
I
actor_network_state/10?-
actor_network_state/1??????????Y
%__inference_signature_wrapper_94457240?

? 
? "?

int64?
int64 	=
%__inference_signature_wrapper_9445728?

? 
? "? 