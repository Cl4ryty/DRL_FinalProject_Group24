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
-adversary_agent/ValueRnnNetwork/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-adversary_agent/ValueRnnNetwork/dense_11/bias
?
Aadversary_agent/ValueRnnNetwork/dense_11/bias/Read/ReadVariableOpReadVariableOp-adversary_agent/ValueRnnNetwork/dense_11/bias*
_output_shapes
:*
dtype0
?
/adversary_agent/ValueRnnNetwork/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*@
shared_name1/adversary_agent/ValueRnnNetwork/dense_11/kernel
?
Cadversary_agent/ValueRnnNetwork/dense_11/kernel/Read/ReadVariableOpReadVariableOp/adversary_agent/ValueRnnNetwork/dense_11/kernel*
_output_shapes

:(*
dtype0
?
Eadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*V
shared_nameGEadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias
?
Yadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias/Read/ReadVariableOpReadVariableOpEadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias*
_output_shapes	
:?*
dtype0
?
Qadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*b
shared_nameSQadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel
?
eadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel/Read/ReadVariableOpReadVariableOpQadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel*
_output_shapes
:	(?*
dtype0
?
Gadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*X
shared_nameIGadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel
?
[adversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel/Read/ReadVariableOpReadVariableOpGadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel*
_output_shapes
:	 ?*
dtype0
?
Madversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias
?
aadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOpReadVariableOpMadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias*
_output_shapes
: *
dtype0
?
Oadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *`
shared_nameQOadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel
?
cadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOpReadVariableOpOadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel*
_output_shapes

:  *
dtype0
?
Ladversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *]
shared_nameNLadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias
?
`adversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias/Read/ReadVariableOpReadVariableOpLadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias*
_output_shapes
: *
dtype0
?
Nadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:M *_
shared_namePNadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel
?
badversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel/Read/ReadVariableOpReadVariableOpNadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel*
_output_shapes

:M *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0
?
Tadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*e
shared_nameVTadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
?
hadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpReadVariableOpTadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
_output_shapes
:*
dtype0
?
Vadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*g
shared_nameXVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
?
jadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpReadVariableOpVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel*
_output_shapes
:	?*
dtype0
?
]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*n
shared_name_]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias
?
qadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias/Read/ReadVariableOpReadVariableOp]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias*
_output_shapes	
:?*
dtype0
?
iadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*z
shared_namekiadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel
?
}adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel/Read/ReadVariableOpReadVariableOpiadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*p
shared_namea_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel
?
sadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel/Read/ReadVariableOpReadVariableOp_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel*
_output_shapes
:	 ?*
dtype0
?
dadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *u
shared_namefdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias
?
xadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias/Read/ReadVariableOpReadVariableOpdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias*
_output_shapes
: *
dtype0
?
fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *w
shared_namehfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel
?
zadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel/Read/ReadVariableOpReadVariableOpfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel*
_output_shapes

:  *
dtype0
?
dadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *u
shared_namefdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias
?
xadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias/Read/ReadVariableOpReadVariableOpdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias*
_output_shapes
: *
dtype0
?
fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:M *w
shared_namehfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel
?
zadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel/Read/ReadVariableOpReadVariableOpfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel*
_output_shapes

:M *
dtype0
v
conv2d_1/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/bias_1
o
#conv2d_1/bias_1/Read/ReadVariableOpReadVariableOpconv2d_1/bias_1*
_output_shapes
:*
dtype0
?
conv2d_1/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_1/kernel_1

%conv2d_1/kernel_1/Read/ReadVariableOpReadVariableOpconv2d_1/kernel_1*&
_output_shapes
:*
dtype0
t
dense_6/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias_1
m
"dense_6/bias_1/Read/ReadVariableOpReadVariableOpdense_6/bias_1*
_output_shapes
:*
dtype0
|
dense_6/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_6/kernel_1
u
$dense_6/kernel_1/Read/ReadVariableOpReadVariableOpdense_6/kernel_1*
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
VP
VARIABLE_VALUEdense_6/kernel_1,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_6/bias_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv2d_1/kernel_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv2d_1/bias_1,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEiadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUETadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_6/kernel-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_6/bias-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_1/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv2d_1/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUENadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUELadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEOadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEMadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEQadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEEadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias-model_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE/adversary_agent/ValueRnnNetwork/dense_11/kernel-model_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE-adversary_agent/ValueRnnNetwork/dense_11/bias-model_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_directionaction_0_observation_imageaction_0_rewardaction_0_step_typeaction_1_actor_network_state_0action_1_actor_network_state_1dense_6/kernel_1dense_6/bias_1conv2d_1/kernel_1conv2d_1/bias_1fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kerneldadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/biasfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kerneldadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kerneliadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/biasVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelTadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
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
GPU 2J 8? */
f*R(
&__inference_signature_wrapper_24445114
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
GPU 2J 8? */
f*R(
&__inference_signature_wrapper_24445123
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
GPU 2J 8? */
f*R(
&__inference_signature_wrapper_24445135
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
GPU 2J 8? */
f*R(
&__inference_signature_wrapper_24445131
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameglobal_step/Read/ReadVariableOp$dense_6/kernel_1/Read/ReadVariableOp"dense_6/bias_1/Read/ReadVariableOp%conv2d_1/kernel_1/Read/ReadVariableOp#conv2d_1/bias_1/Read/ReadVariableOpzadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel/Read/ReadVariableOpxadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias/Read/ReadVariableOpzadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel/Read/ReadVariableOpxadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias/Read/ReadVariableOpsadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel/Read/ReadVariableOp}adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel/Read/ReadVariableOpqadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias/Read/ReadVariableOpjadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOphadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOpbadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel/Read/ReadVariableOp`adversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias/Read/ReadVariableOpcadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOpaadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOp[adversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel/Read/ReadVariableOpeadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel/Read/ReadVariableOpYadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias/Read/ReadVariableOpCadversary_agent/ValueRnnNetwork/dense_11/kernel/Read/ReadVariableOpAadversary_agent/ValueRnnNetwork/dense_11/bias/Read/ReadVariableOpConst*(
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
GPU 2J 8? **
f%R#
!__inference__traced_save_24446255
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameglobal_stepdense_6/kernel_1dense_6/bias_1conv2d_1/kernel_1conv2d_1/bias_1fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kerneldadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/biasfadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kerneldadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kerneliadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/biasVadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelTadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasdense_6/kerneldense_6/biasconv2d_1/kernelconv2d_1/biasNadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernelLadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/biasOadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernelMadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/biasGadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernelQadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernelEadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias/adversary_agent/ValueRnnNetwork/dense_11/kernel-adversary_agent/ValueRnnNetwork/dense_11/bias*'
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_24446346Ց
?
-
+__inference_function_with_signature_1946645?
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
GPU 2J 8? *"
fR
__inference_<lambda>_2179*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445460

inputs"
dense_6_24445454:
dense_6_24445456:
identity??dense_6/StatefulPartitionedCall?
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445441?
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445454dense_6_24445456*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445453w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_re_lu_1_layer_call_fn_24446123

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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445592h
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
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445979

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
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446026

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
?
?
+__inference_function_with_signature_1946551
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
__inference_action_1946518k
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
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_24446075

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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445673

inputs+
conv2d_1_24445665:
conv2d_1_24445667:
identity?? conv2d_1/StatefulPartitionedCall?
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445647?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445665conv2d_1_24445667*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445581?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445592?
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445600q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445152

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
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445747

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp^
lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
lambda_3/one_hotOneHotinputslambda_3/one_hot/depth:output:0"lambda_3/one_hot/on_value:output:0#lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????g
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lambda_3/ReshapeReshapelambda_3/one_hot:output:0lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_6/MatMulMatMullambda_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445555
lambda_3_input"
dense_6_24445549:
dense_6_24445551:
identity??dense_6/StatefulPartitionedCall?
lambda_3/PartitionedCallPartitionedCalllambda_3_input*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445495?
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445549dense_6_24445551*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445453w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_3_input
?
?
/__inference_sequential_2_layer_call_fn_24445881

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
GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445673o
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
/__inference_sequential_2_layer_call_fn_24445689
lambda_2_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0*
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
GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445673o
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
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_2_input
?
G
+__inference_lambda_2_layer_call_fn_24445972

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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445358h
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
?
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446128

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
?
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445311

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
G
+__inference_lambda_3_layer_call_fn_24446031

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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445441`
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
?
+__inference_conv2d_1_layer_call_fn_24445995

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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445292w
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
?
`
__inference_<lambda>_2176!
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
?
?
/__inference_sequential_3_layer_call_fn_24445822

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
GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445460o
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
??
?
__inference_action_1947119
time_step_step_type
time_step_reward
time_step_discount#
time_step_observation_direction
time_step_observation_image&
"policy_state_actor_network_state_0&
"policy_state_actor_network_state_1?
{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource:?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource:?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource: ?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource: ?
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource:	 ??
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource:
???
tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource:	?q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	?m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2??fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp?rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp?kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp?jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp?lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpG
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
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:??????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape:output:0zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul:product:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????H?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concatConcatV2mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd:output:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Reshape:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????M?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M   ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????M?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
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
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
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
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/RankConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/delta:output:0*
_output_shapes
:?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat:output:0*
T0*+
_output_shapes
:????????? ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*
_output_shapes
:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims:output:0*
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

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOps^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp2?
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOprActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp2?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp2?
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp2?
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp2?
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
?
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446092

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
Z

__inference_<lambda>_2179*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445412
lambda_2_input+
conv2d_1_24445404:
conv2d_1_24445406:
identity?? conv2d_1/StatefulPartitionedCall?
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445280?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445404conv2d_1_24445406*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445292?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445303?
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445311q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_2_input
?
[
+__inference_function_with_signature_1946618

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
%__inference_get_initial_state_1946613a
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
?
(
&__inference_signature_wrapper_24445135?
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
+__inference_function_with_signature_1946645*(
_construction_contextkEagerRuntime*
_input_shapes 
?
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446099

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
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_24445164

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
?
k
+__inference_function_with_signature_1946634
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
GPU 2J 8? *"
fR
__inference_<lambda>_2176^
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
?
?
/__inference_sequential_2_layer_call_fn_24445610
lambda_2_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0*
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
GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445603o
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
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_2_input
?
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445441

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
/__inference_sequential_3_layer_call_fn_24445722

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
GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445171o
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
?
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445986

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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445797

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOpf
lambda_2/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????W
lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
lambda_2/truedivRealDivlambda_2/Cast:y:0lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dlambda_2/truediv:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
flatten_3/ReshapeReshapere_lu_1/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????Hi
IdentityIdentityflatten_3/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_2_layer_call_fn_24445872

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
GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445603o
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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445713
lambda_2_input+
conv2d_1_24445705:
conv2d_1_24445707:
identity?? conv2d_1/StatefulPartitionedCall?
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445647?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445705conv2d_1_24445707*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445581?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445592?
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445600q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_2_input
?
?
*__inference_dense_6_layer_call_fn_24445952

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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445164o
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
?
V
&__inference_signature_wrapper_24445123

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
+__inference_function_with_signature_1946618a
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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445424
lambda_2_input+
conv2d_1_24445416:
conv2d_1_24445418:
identity?? conv2d_1/StatefulPartitionedCall?
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445358?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445416conv2d_1_24445418*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445292?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445303?
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445311q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_2_input
?
?
/__inference_sequential_3_layer_call_fn_24445731

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
GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445230o
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
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445592

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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445314

inputs+
conv2d_1_24445293:
conv2d_1_24445295:
identity?? conv2d_1/StatefulPartitionedCall?
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445280?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445293conv2d_1_24445295*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445292?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445303?
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445311q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_3_layer_call_fn_24445831

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
GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445519o
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
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445206

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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445292

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
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445545
lambda_3_input"
dense_6_24445539:
dense_6_24445541:
identity??dense_6/StatefulPartitionedCall?
lambda_3/PartitionedCallPartitionedCalllambda_3_input*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445441?
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445539dense_6_24445541*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445453w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_3_input
?
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24446056

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
+__inference_conv2d_1_layer_call_fn_24446108

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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445581w
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
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_24445962

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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445897

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOpf
lambda_2/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????W
lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
lambda_2/truedivRealDivlambda_2/Cast:y:0lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dlambda_2/truediv:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
flatten_3/ReshapeReshapere_lu_1/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????Hi
IdentityIdentityflatten_3/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446118

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
?
U
%__inference_get_initial_state_1946613

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
?
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445358

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
G
+__inference_lambda_3_layer_call_fn_24445923

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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445206`
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445266
lambda_3_input"
dense_6_24445260:
dense_6_24445262:
identity??dense_6/StatefulPartitionedCall?
lambda_3/PartitionedCallPartitionedCalllambda_3_input*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445206?
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445260dense_6_24445262*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445164w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_3_input
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445863

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp^
lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
lambda_3/one_hotOneHotinputslambda_3/one_hot/depth:output:0"lambda_3/one_hot/on_value:output:0#lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????g
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lambda_3/ReshapeReshapelambda_3/one_hot:output:0lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_6/MatMulMatMullambda_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_2_layer_call_fn_24446085

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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445647h
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
?K
?
!__inference__traced_save_24446255
file_prefix*
&savev2_global_step_read_readvariableop	/
+savev2_dense_6_kernel_1_read_readvariableop-
)savev2_dense_6_bias_1_read_readvariableop0
,savev2_conv2d_1_kernel_1_read_readvariableop.
*savev2_conv2d_1_bias_1_read_readvariableop?
?savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_kernel_read_readvariableop?
savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_bias_read_readvariableop?
?savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_kernel_read_readvariableop?
savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_bias_read_readvariableop~
zsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_kernel_read_readvariableop?
?savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_recurrent_kernel_read_readvariableop|
xsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_bias_read_readvariableopu
qsavev2_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableops
osavev2_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableopm
isavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_kernel_read_readvariableopk
gsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_bias_read_readvariableopn
jsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_kernel_read_readvariableopl
hsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_bias_read_readvariableopf
bsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_kernel_read_readvariableopp
lsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_recurrent_kernel_read_readvariableopd
`savev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_bias_read_readvariableopN
Jsavev2_adversary_agent_valuernnnetwork_dense_11_kernel_read_readvariableopL
Hsavev2_adversary_agent_valuernnnetwork_dense_11_bias_read_readvariableop
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
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_global_step_read_readvariableop+savev2_dense_6_kernel_1_read_readvariableop)savev2_dense_6_bias_1_read_readvariableop,savev2_conv2d_1_kernel_1_read_readvariableop*savev2_conv2d_1_bias_1_read_readvariableop?savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_kernel_read_readvariableopsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_bias_read_readvariableop?savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_kernel_read_readvariableopsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_bias_read_readvariableopzsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_kernel_read_readvariableop?savev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_recurrent_kernel_read_readvariableopxsavev2_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_bias_read_readvariableopqsavev2_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableoposavev2_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableopisavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_kernel_read_readvariableopgsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_bias_read_readvariableopjsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_kernel_read_readvariableophsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_bias_read_readvariableopbsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_kernel_read_readvariableoplsavev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_recurrent_kernel_read_readvariableop`savev2_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_bias_read_readvariableopJsavev2_adversary_agent_valuernnnetwork_dense_11_kernel_read_readvariableopHsavev2_adversary_agent_valuernnnetwork_dense_11_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446139

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
?
?
/__inference_sequential_2_layer_call_fn_24445400
lambda_2_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0*
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
GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445384o
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
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_2_input
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445847

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp^
lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
lambda_3/one_hotOneHotinputslambda_3/one_hot/depth:output:0"lambda_3/one_hot/on_value:output:0#lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????g
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lambda_3/ReshapeReshapelambda_3/one_hot:output:0lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_6/MatMulMatMullambda_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
#__inference_distribution_fn_1947329
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1?
{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource:?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource:?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource: ?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource: ?
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource:	 ??
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource:
???
tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource:	?q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	?m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1	

identity_2	

identity_3

identity_4??fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp?rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp?kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp?jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp?lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:??????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape:output:0zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul:product:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????H?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concatConcatV2mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd:output:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Reshape:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????M?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M   ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????M?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
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
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
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
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/RankConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/delta:output:0*
_output_shapes
:?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat:output:0*
T0*+
_output_shapes
:????????? ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*
_output_shapes
:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims:output:0*
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

Identity_3Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_4Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOps^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
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
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp2?
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOprActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp2?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp2?
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp2?
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp2?
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
?	
?
E__inference_dense_6_layer_call_and_return_conditional_losses_24445453

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
?
?
/__inference_sequential_3_layer_call_fn_24445467
lambda_3_input
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0*
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
GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445460o
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
_user_specified_namelambda_3_input
?
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445600

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
?
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445933

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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445171

inputs"
dense_6_24445165:
dense_6_24445167:
identity??dense_6/StatefulPartitionedCall?
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445152?
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445165dense_6_24445167*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445164w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_24445114
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
+__inference_function_with_signature_1946551k
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
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445256
lambda_3_input"
dense_6_24445250:
dense_6_24445252:
identity??dense_6/StatefulPartitionedCall?
lambda_3/PartitionedCallPartitionedCalllambda_3_input*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445152?
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445250dense_6_24445252*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445164w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_3_input
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445763

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp^
lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    X
lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
lambda_3/one_hotOneHotinputslambda_3/one_hot/depth:output:0"lambda_3/one_hot/on_value:output:0#lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????g
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lambda_3/ReshapeReshapelambda_3/one_hot:output:0lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_6/MatMulMatMullambda_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446005

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
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445943

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
G
+__inference_lambda_2_layer_call_fn_24445967

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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445280h
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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445813

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOpf
lambda_2/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????W
lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
lambda_2/truedivRealDivlambda_2/Cast:y:0lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dlambda_2/truediv:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
flatten_3/ReshapeReshapere_lu_1/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????Hi
IdentityIdentityflatten_3/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_6_layer_call_fn_24446065

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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445453o
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
/__inference_sequential_3_layer_call_fn_24445535
lambda_3_input
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0*
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
GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445519o
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
_user_specified_namelambda_3_input
?
F
*__inference_re_lu_1_layer_call_fn_24446010

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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445303h
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
U
%__inference_get_initial_state_1947345

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
?
H
,__inference_flatten_3_layer_call_fn_24446020

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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445311`
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
?
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24446046

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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445384

inputs+
conv2d_1_24445376:
conv2d_1_24445378:
identity?? conv2d_1/StatefulPartitionedCall?
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445358?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445376conv2d_1_24445378*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445292?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445303?
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445311q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_2_layer_call_fn_24445772

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
GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445314o
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
/__inference_sequential_3_layer_call_fn_24445246
lambda_3_input
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0*
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
GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445230o
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
_user_specified_namelambda_3_input
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445230

inputs"
dense_6_24445224:
dense_6_24445226:
identity??dense_6/StatefulPartitionedCall?
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445206?
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445224dense_6_24445226*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445164w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445280

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
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445569

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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445701
lambda_2_input+
conv2d_1_24445693:
conv2d_1_24445695:
identity?? conv2d_1/StatefulPartitionedCall?
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445569?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445693conv2d_1_24445695*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445581?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445592?
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445600q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_2_input
??
?
__inference_action_1946884
	step_type

reward
discount
observation_direction
observation_image
actor_network_state_0
actor_network_state_1?
{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource:?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource:?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource: ?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource: ?
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource:	 ??
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource:
???
tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource:	?q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	?m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2??fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp?rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp?kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp?jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp?lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:??????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape:output:0zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul:product:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????H?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concatConcatV2mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd:output:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Reshape:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????M?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M   ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????M?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
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
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
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
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/RankConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/delta:output:0*
_output_shapes
:?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat:output:0*
T0*+
_output_shapes
:????????? ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*
_output_shapes
:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims:output:0*
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

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOps^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp2?
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOprActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp2?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp2?
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp2?
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp2?
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
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445519

inputs"
dense_6_24445513:
dense_6_24445515:
identity??dense_6/StatefulPartitionedCall?
lambda_3/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445495?
dense_6/StatefulPartitionedCallStatefulPartitionedCall!lambda_3/PartitionedCall:output:0dense_6_24445513dense_6_24445515*
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
GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_24445453w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445647

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
f
&__inference_signature_wrapper_24445131
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
+__inference_function_with_signature_1946634^
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
?
?
/__inference_sequential_3_layer_call_fn_24445178
lambda_3_input
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_3_inputunknown	unknown_0*
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
GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445171o
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
_user_specified_namelambda_3_input
?

?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445581

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
?
/__inference_sequential_2_layer_call_fn_24445781

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
GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445384o
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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445603

inputs+
conv2d_1_24445582:
conv2d_1_24445584:
identity?? conv2d_1/StatefulPartitionedCall?
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445569?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0conv2d_1_24445582conv2d_1_24445584*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24445581?
re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445592?
flatten_3/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445600q
IdentityIdentity"flatten_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hi
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445495

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
G
+__inference_lambda_3_layer_call_fn_24445918

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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445152`
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
?
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445913

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOpf
lambda_2/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????W
lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
lambda_2/truedivRealDivlambda_2/Cast:y:0lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dlambda_2/truediv:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
flatten_3/ReshapeReshapere_lu_1/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????Hi
IdentityIdentityflatten_3/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_3_layer_call_fn_24446036

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
GPU 2J 8? *O
fJRH
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445495`
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
ߖ
?
__inference_action_1946518
	time_step
time_step_1
time_step_2
time_step_3
time_step_4
policy_state
policy_state_1?
{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource:?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource:?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource:M }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource: ?
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:  }
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource: ?
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource:	 ??
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource:
???
tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource:	?q
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:	?m
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:
identity	

identity_1

identity_2??fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp?eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp?rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp?kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp?jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp?lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp@
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
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hotOneHotfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:??????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp{actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/lambda_3/Reshape:output:0zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul:product:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/CastCasthActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv/y:output:0*
T0*/
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/lambda_2/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H   ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/re_lu_1/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????H?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concatConcatV2mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd:output:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/flatten_3/Reshape:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????M?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????M   ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_1/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????M?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_matmul_readvariableop_resource*
_output_shapes

:M *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_4/Reshape:output:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
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
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
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
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/RankConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range/delta:output:0*
_output_shapes
:?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/concat:output:0*
T0*+
_output_shapes
:????????? ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*
_output_shapes
:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/Select_1:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/ExpandDims:output:0*
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

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOps^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/MatMul/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_2/conv2d_1/Conv2D/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/BiasAdd/ReadVariableOp2?
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOprActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_3/dense_6/MatMul/ReadVariableOp2?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/BiasAdd/ReadVariableOp2?
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul/ReadVariableOp2?
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/lstm_cell_2/MatMul_1/ReadVariableOp2?
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
?
G
+__inference_lambda_2_layer_call_fn_24446080

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
GPU 2J 8? *O
fJRH
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445569h
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
?
H
,__inference_flatten_3_layer_call_fn_24446133

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
GPU 2J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_24445600`
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
?{
?
$__inference__traced_restore_24446346
file_prefix&
assignvariableop_global_step:	 5
#assignvariableop_1_dense_6_kernel_1:/
!assignvariableop_2_dense_6_bias_1:>
$assignvariableop_3_conv2d_1_kernel_1:0
"assignvariableop_4_conv2d_1_bias_1:?
yassignvariableop_5_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_kernel:M ?
wassignvariableop_6_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_bias: ?
yassignvariableop_7_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_kernel:  ?
wassignvariableop_8_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_bias: ?
rassignvariableop_9_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_kernel:	 ??
}assignvariableop_10_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_recurrent_kernel:
???
qassignvariableop_11_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_bias:	?}
jassignvariableop_12_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel:	?v
hassignvariableop_13_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias:4
"assignvariableop_14_dense_6_kernel:.
 assignvariableop_15_dense_6_bias:=
#assignvariableop_16_conv2d_1_kernel:/
!assignvariableop_17_conv2d_1_bias:t
bassignvariableop_18_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_kernel:M n
`assignvariableop_19_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_bias: u
cassignvariableop_20_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_kernel:  o
aassignvariableop_21_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_bias: n
[assignvariableop_22_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_kernel:	 ?x
eassignvariableop_23_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_recurrent_kernel:	(?h
Yassignvariableop_24_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_bias:	?U
Cassignvariableop_25_adversary_agent_valuernnnetwork_dense_11_kernel:(O
Aassignvariableop_26_adversary_agent_valuernnnetwork_dense_11_bias:
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
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_6_kernel_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_6_bias_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv2d_1_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpyassignvariableop_5_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpwassignvariableop_6_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_7_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpyassignvariableop_7_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpwassignvariableop_8_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_8_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOprassignvariableop_9_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp}assignvariableop_10_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpqassignvariableop_11_adversary_agent_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpjassignvariableop_12_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOphassignvariableop_13_adversary_agent_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpbassignvariableop_18_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp`assignvariableop_19_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpcassignvariableop_20_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpaassignvariableop_21_adversary_agent_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp[assignvariableop_22_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpeassignvariableop_23_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_recurrent_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpYassignvariableop_24_adversary_agent_valuernnnetwork_valuernnnetwork_dynamic_unroll_3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpCassignvariableop_25_adversary_agent_valuernnnetwork_dense_11_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adversary_agent_valuernnnetwork_dense_11_biasIdentity_26:output:0"/device:CPU:0*
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
?
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446015

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
/__inference_sequential_2_layer_call_fn_24445321
lambda_2_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0*
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
GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445314o
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
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_2_input
?
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24445303

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
__inference_action_1946884
__inference_action_1947119?
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
#__inference_distribution_fn_1947329?
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
%__inference_get_initial_state_1947345?
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
__inference_<lambda>_2179"?
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
__inference_<lambda>_2176"?
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
 :2dense_6/kernel
:2dense_6/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
x:vM 2fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/kernel
r:p 2dadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_7/bias
x:v  2fadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/kernel
r:p 2dadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_8/bias
r:p	 ?2_adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/kernel
}:{
??2iadversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/recurrent_kernel
l:j?2]adversary_agent/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_2/bias
i:g	?2Vadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
b:`2Tadversary_agent/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
 :2dense_6/kernel
:2dense_6/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
`:^M 2Nadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/kernel
Z:X 2Ladversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_9/bias
a:_  2Oadversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/kernel
[:Y 2Madversary_agent/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_10/bias
Z:X	 ?2Gadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/kernel
d:b	(?2Qadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/recurrent_kernel
T:R?2Eadversary_agent/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_3/bias
A:?(2/adversary_agent/ValueRnnNetwork/dense_11/kernel
;:92-adversary_agent/ValueRnnNetwork/dense_11/bias
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
__inference_action_1946884	step_typerewarddiscountobservation/directionobservation/imageactor_network_state/0actor_network_state/1"?
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
__inference_action_1947119time_step/step_typetime_step/rewardtime_step/discounttime_step/observation/directiontime_step/observation/image"policy_state/actor_network_state/0"policy_state/actor_network_state/1"?
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
#__inference_distribution_fn_1947329	step_typerewarddiscountobservation/directionobservation/imageactor_network_state/0actor_network_state/1"?
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
%__inference_get_initial_state_1947345
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
&__inference_signature_wrapper_24445114
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
&__inference_signature_wrapper_24445123
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
&__inference_signature_wrapper_24445131"?
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
&__inference_signature_wrapper_24445135"?
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
/__inference_sequential_3_layer_call_fn_24445178
/__inference_sequential_3_layer_call_fn_24445722
/__inference_sequential_3_layer_call_fn_24445731
/__inference_sequential_3_layer_call_fn_24445246?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445747
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445763
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445256
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445266?
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
?trace_32?
/__inference_sequential_2_layer_call_fn_24445321
/__inference_sequential_2_layer_call_fn_24445772
/__inference_sequential_2_layer_call_fn_24445781
/__inference_sequential_2_layer_call_fn_24445400?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445797
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445813
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445412
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445424?
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
/__inference_sequential_3_layer_call_fn_24445467
/__inference_sequential_3_layer_call_fn_24445822
/__inference_sequential_3_layer_call_fn_24445831
/__inference_sequential_3_layer_call_fn_24445535?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445847
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445863
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445545
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445555?
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
?trace_32?
/__inference_sequential_2_layer_call_fn_24445610
/__inference_sequential_2_layer_call_fn_24445872
/__inference_sequential_2_layer_call_fn_24445881
/__inference_sequential_2_layer_call_fn_24445689?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445897
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445913
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445701
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445713?
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
+__inference_lambda_3_layer_call_fn_24445918
+__inference_lambda_3_layer_call_fn_24445923?
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445933
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445943?
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
*__inference_dense_6_layer_call_fn_24445952?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_24445962?
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
/__inference_sequential_3_layer_call_fn_24445178lambda_3_input"?
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
/__inference_sequential_3_layer_call_fn_24445722inputs"?
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
/__inference_sequential_3_layer_call_fn_24445731inputs"?
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
/__inference_sequential_3_layer_call_fn_24445246lambda_3_input"?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445747inputs"?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445763inputs"?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445256lambda_3_input"?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445266lambda_3_input"?
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
+__inference_lambda_2_layer_call_fn_24445967
+__inference_lambda_2_layer_call_fn_24445972?
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445979
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445986?
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
+__inference_conv2d_1_layer_call_fn_24445995?
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446005?
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
*__inference_re_lu_1_layer_call_fn_24446010?
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
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446015?
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
,__inference_flatten_3_layer_call_fn_24446020?
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446026?
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
/__inference_sequential_2_layer_call_fn_24445321lambda_2_input"?
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
/__inference_sequential_2_layer_call_fn_24445772inputs"?
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
/__inference_sequential_2_layer_call_fn_24445781inputs"?
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
/__inference_sequential_2_layer_call_fn_24445400lambda_2_input"?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445797inputs"?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445813inputs"?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445412lambda_2_input"?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445424lambda_2_input"?
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
+__inference_lambda_3_layer_call_fn_24446031
+__inference_lambda_3_layer_call_fn_24446036?
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24446046
F__inference_lambda_3_layer_call_and_return_conditional_losses_24446056?
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
*__inference_dense_6_layer_call_fn_24446065?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_24446075?
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
/__inference_sequential_3_layer_call_fn_24445467lambda_3_input"?
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
/__inference_sequential_3_layer_call_fn_24445822inputs"?
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
/__inference_sequential_3_layer_call_fn_24445831inputs"?
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
/__inference_sequential_3_layer_call_fn_24445535lambda_3_input"?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445847inputs"?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445863inputs"?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445545lambda_3_input"?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445555lambda_3_input"?
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
+__inference_lambda_2_layer_call_fn_24446080
+__inference_lambda_2_layer_call_fn_24446085?
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446092
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446099?
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
+__inference_conv2d_1_layer_call_fn_24446108?
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446118?
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
*__inference_re_lu_1_layer_call_fn_24446123?
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
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446128?
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
,__inference_flatten_3_layer_call_fn_24446133?
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446139?
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
/__inference_sequential_2_layer_call_fn_24445610lambda_2_input"?
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
/__inference_sequential_2_layer_call_fn_24445872inputs"?
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
/__inference_sequential_2_layer_call_fn_24445881inputs"?
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
/__inference_sequential_2_layer_call_fn_24445689lambda_2_input"?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445897inputs"?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445913inputs"?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445701lambda_2_input"?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445713lambda_2_input"?
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
+__inference_lambda_3_layer_call_fn_24445918inputs"?
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
+__inference_lambda_3_layer_call_fn_24445923inputs"?
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445933inputs"?
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445943inputs"?
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
*__inference_dense_6_layer_call_fn_24445952inputs"?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_24445962inputs"?
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
+__inference_lambda_2_layer_call_fn_24445967inputs"?
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
+__inference_lambda_2_layer_call_fn_24445972inputs"?
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445979inputs"?
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445986inputs"?
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
+__inference_conv2d_1_layer_call_fn_24445995inputs"?
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446005inputs"?
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
*__inference_re_lu_1_layer_call_fn_24446010inputs"?
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
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446015inputs"?
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
,__inference_flatten_3_layer_call_fn_24446020inputs"?
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446026inputs"?
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
+__inference_lambda_3_layer_call_fn_24446031inputs"?
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
+__inference_lambda_3_layer_call_fn_24446036inputs"?
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24446046inputs"?
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24446056inputs"?
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
*__inference_dense_6_layer_call_fn_24446065inputs"?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_24446075inputs"?
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
+__inference_lambda_2_layer_call_fn_24446080inputs"?
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
+__inference_lambda_2_layer_call_fn_24446085inputs"?
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446092inputs"?
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446099inputs"?
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
+__inference_conv2d_1_layer_call_fn_24446108inputs"?
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446118inputs"?
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
*__inference_re_lu_1_layer_call_fn_24446123inputs"?
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
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446128inputs"?
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
,__inference_flatten_3_layer_call_fn_24446133inputs"?
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446139inputs"?
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
 8
__inference_<lambda>_2176?

? 
? "? 	1
__inference_<lambda>_2179?

? 
? "? ?
__inference_action_1946884????
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
__inference_action_1947119????
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446005l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_24446118l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_1_layer_call_fn_24445995_7?4
-?*
(?%
inputs?????????
? " ???????????
+__inference_conv2d_1_layer_call_fn_24446108_7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_dense_6_layer_call_and_return_conditional_losses_24445962\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
E__inference_dense_6_layer_call_and_return_conditional_losses_24446075\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_6_layer_call_fn_24445952O/?,
%?"
 ?
inputs?????????
? "??????????}
*__inference_dense_6_layer_call_fn_24446065O/?,
%?"
 ?
inputs?????????
? "???????????
#__inference_distribution_fn_1947329????
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446026`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????H
? ?
G__inference_flatten_3_layer_call_and_return_conditional_losses_24446139`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????H
? ?
,__inference_flatten_3_layer_call_fn_24446020S7?4
-?*
(?%
inputs?????????
? "??????????H?
,__inference_flatten_3_layer_call_fn_24446133S7?4
-?*
(?%
inputs?????????
? "??????????H?
%__inference_get_initial_state_1947345?"?
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445979p??<
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24445986p??<
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446092p??<
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
F__inference_lambda_2_layer_call_and_return_conditional_losses_24446099p??<
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
+__inference_lambda_2_layer_call_fn_24445967c??<
5?2
(?%
inputs?????????

 
p 
? " ???????????
+__inference_lambda_2_layer_call_fn_24445972c??<
5?2
(?%
inputs?????????

 
p
? " ???????????
+__inference_lambda_2_layer_call_fn_24446080c??<
5?2
(?%
inputs?????????

 
p 
? " ???????????
+__inference_lambda_2_layer_call_fn_24446085c??<
5?2
(?%
inputs?????????

 
p
? " ???????????
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445933`7?4
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24445943`7?4
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24446046`7?4
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
F__inference_lambda_3_layer_call_and_return_conditional_losses_24446056`7?4
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
+__inference_lambda_3_layer_call_fn_24445918S7?4
-?*
 ?
inputs?????????

 
p 
? "???????????
+__inference_lambda_3_layer_call_fn_24445923S7?4
-?*
 ?
inputs?????????

 
p
? "???????????
+__inference_lambda_3_layer_call_fn_24446031S7?4
-?*
 ?
inputs?????????

 
p 
? "???????????
+__inference_lambda_3_layer_call_fn_24446036S7?4
-?*
 ?
inputs?????????

 
p
? "???????????
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446015h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
E__inference_re_lu_1_layer_call_and_return_conditional_losses_24446128h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_re_lu_1_layer_call_fn_24446010[7?4
-?*
(?%
inputs?????????
? " ???????????
*__inference_re_lu_1_layer_call_fn_24446123[7?4
-?*
(?%
inputs?????????
? " ???????????
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445412tG?D
=?:
0?-
lambda_2_input?????????
p 

 
? "%?"
?
0?????????H
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445424tG?D
=?:
0?-
lambda_2_input?????????
p

 
? "%?"
?
0?????????H
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445701tG?D
=?:
0?-
lambda_2_input?????????
p 

 
? "%?"
?
0?????????H
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445713tG?D
=?:
0?-
lambda_2_input?????????
p

 
? "%?"
?
0?????????H
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445797l??<
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445813l??<
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445897l??<
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_24445913l??<
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
/__inference_sequential_2_layer_call_fn_24445321gG?D
=?:
0?-
lambda_2_input?????????
p 

 
? "??????????H?
/__inference_sequential_2_layer_call_fn_24445400gG?D
=?:
0?-
lambda_2_input?????????
p

 
? "??????????H?
/__inference_sequential_2_layer_call_fn_24445610gG?D
=?:
0?-
lambda_2_input?????????
p 

 
? "??????????H?
/__inference_sequential_2_layer_call_fn_24445689gG?D
=?:
0?-
lambda_2_input?????????
p

 
? "??????????H?
/__inference_sequential_2_layer_call_fn_24445772_??<
5?2
(?%
inputs?????????
p 

 
? "??????????H?
/__inference_sequential_2_layer_call_fn_24445781_??<
5?2
(?%
inputs?????????
p

 
? "??????????H?
/__inference_sequential_2_layer_call_fn_24445872_??<
5?2
(?%
inputs?????????
p 

 
? "??????????H?
/__inference_sequential_2_layer_call_fn_24445881_??<
5?2
(?%
inputs?????????
p

 
? "??????????H?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445256l??<
5?2
(?%
lambda_3_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445266l??<
5?2
(?%
lambda_3_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445545l??<
5?2
(?%
lambda_3_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445555l??<
5?2
(?%
lambda_3_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445747d7?4
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445763d7?4
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445847d7?4
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_24445863d7?4
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
/__inference_sequential_3_layer_call_fn_24445178_??<
5?2
(?%
lambda_3_input?????????
p 

 
? "???????????
/__inference_sequential_3_layer_call_fn_24445246_??<
5?2
(?%
lambda_3_input?????????
p

 
? "???????????
/__inference_sequential_3_layer_call_fn_24445467_??<
5?2
(?%
lambda_3_input?????????
p 

 
? "???????????
/__inference_sequential_3_layer_call_fn_24445535_??<
5?2
(?%
lambda_3_input?????????
p

 
? "???????????
/__inference_sequential_3_layer_call_fn_24445722W7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
/__inference_sequential_3_layer_call_fn_24445731W7?4
-?*
 ?
inputs?????????
p

 
? "???????????
/__inference_sequential_3_layer_call_fn_24445822W7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
/__inference_sequential_3_layer_call_fn_24445831W7?4
-?*
 ?
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_24445114????
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
&__inference_signature_wrapper_24445123?0?-
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
actor_network_state/1??????????Z
&__inference_signature_wrapper_244451310?

? 
? "?

int64?
int64 	>
&__inference_signature_wrapper_24445135?

? 
? "? 