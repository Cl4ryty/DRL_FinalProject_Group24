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
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??
?
+adversary_env/ValueRnnNetwork/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+adversary_env/ValueRnnNetwork/dense_17/bias
?
?adversary_env/ValueRnnNetwork/dense_17/bias/Read/ReadVariableOpReadVariableOp+adversary_env/ValueRnnNetwork/dense_17/bias*
_output_shapes
:*
dtype0
?
-adversary_env/ValueRnnNetwork/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*>
shared_name/-adversary_env/ValueRnnNetwork/dense_17/kernel
?
Aadversary_env/ValueRnnNetwork/dense_17/kernel/Read/ReadVariableOpReadVariableOp-adversary_env/ValueRnnNetwork/dense_17/kernel*
_output_shapes

:(*
dtype0
?
Cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias
?
Wadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias/Read/ReadVariableOpReadVariableOpCadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias*
_output_shapes	
:?*
dtype0
?
Oadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*`
shared_nameQOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel
?
cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel/Read/ReadVariableOpReadVariableOpOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel*
_output_shapes
:	(?*
dtype0
?
Eadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*V
shared_nameGEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel
?
Yadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel/Read/ReadVariableOpReadVariableOpEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel*
_output_shapes
:	 ?*
dtype0
?
Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *\
shared_nameMKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias
?
_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias/Read/ReadVariableOpReadVariableOpKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias*
_output_shapes
: *
dtype0
?
Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *^
shared_nameOMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel
?
aadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel/Read/ReadVariableOpReadVariableOpMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel*
_output_shapes

:  *
dtype0
?
Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *\
shared_nameMKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias
?
_adversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias/Read/ReadVariableOpReadVariableOpKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias*
_output_shapes
: *
dtype0
?
Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *^
shared_nameOMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel
?
aadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel/Read/ReadVariableOpReadVariableOpMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel*
_output_shapes
:	? *
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
?
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
?
Radversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*c
shared_nameTRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
?
fadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpReadVariableOpRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias*
_output_shapes	
:?*
dtype0
?
Tadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*e
shared_nameVTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
?
hadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpReadVariableOpTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel* 
_output_shapes
:
??*
dtype0
?
[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*l
shared_name][adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias
?
oadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias/Read/ReadVariableOpReadVariableOp[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias*
_output_shapes	
:?*
dtype0
?
gadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*x
shared_nameigadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel
?
{adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel/Read/ReadVariableOpReadVariableOpgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*n
shared_name_]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel
?
qadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel/Read/ReadVariableOpReadVariableOp]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel*
_output_shapes
:	 ?*
dtype0
?
cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *t
shared_nameecadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias
?
wadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias/Read/ReadVariableOpReadVariableOpcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias*
_output_shapes
: *
dtype0
?
eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *v
shared_namegeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel
?
yadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel/Read/ReadVariableOpReadVariableOpeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel*
_output_shapes

:  *
dtype0
?
cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *t
shared_nameecadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias
?
wadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOpReadVariableOpcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias*
_output_shapes
: *
dtype0
?
eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *v
shared_namegeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel
?
yadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOpReadVariableOpeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel*
_output_shapes
:	? *
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
?
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
WQ
VARIABLE_VALUEconv2d_2/kernel_1,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv2d_2/bias_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdense_12/kernel_1,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_12/bias_1,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEeadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUETadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_2/kernel-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv2d_2/bias-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdense_12/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_12/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEMadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEEadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEOadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
??
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

?0
?1
?2*
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

?0
?1
?2*
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
<
?0
?1
?2
?3
?4
?5
?6*
* 
* 
* 
?
?layer-0
?layer_with_weights-0
?layer-1
?layer-2
?layer-3
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?layer-0
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
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
<
?0
?1
?2
?3
?4
?5
?6*
* 
* 
* 
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
?
?layer-0
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
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
bias
!?_jit_compiled_convolution_op*
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
+?&call_and_return_all_conditional_losses* 
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

kernel
bias*
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
bias
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
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
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
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
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
* 
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
0
1*

0
1*
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
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
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
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
$
?0
?1
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
* 


?0* 
* 
* 
* 
* 
* 
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
* 
* 
* 
* 
* 
* 
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 


?0* 
* 
* 
* 
* 
* 
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
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
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
action_0_observation_imagePlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
action_0_observation_random_zPlaceholder*'
_output_shapes
:?????????2*
dtype0*
shape:?????????2
?
action_0_observation_time_stepPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
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
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observation_imageaction_0_observation_random_zaction_0_observation_time_stepaction_0_rewardaction_0_step_typeaction_1_actor_network_state_0action_1_actor_network_state_1conv2d_2/kernel_1conv2d_2/bias_1dense_12/kernel_1dense_12/bias_1eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/biaseadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernelcadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernelgadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/biasTadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernelRadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias* 
Tin
2*
Tout
2	*
_collective_manager_ids
 *K
_output_shapes9
7:?????????:??????????:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_24446640
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
&__inference_signature_wrapper_24446649
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
&__inference_signature_wrapper_24446661
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
&__inference_signature_wrapper_24446657
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
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
GPU 2J 8? **
f%R#
!__inference__traced_save_24447978
?
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_24448069??
?
G
+__inference_lambda_4_layer_call_fn_24447604

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24446675h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447783

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_4_layer_call_fn_24447381

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446779p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447227
lambda_6_input
identity?
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447181i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:W S
'
_output_shapes
:?????????2
(
_user_specified_namelambda_6_input
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447353
lambda_5_input#
dense_12_24447347:4

dense_12_24447349:

identity?? dense_12/StatefulPartitionedCall?
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447249?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_24447347dense_12_24447349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24447261x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_5_input
?
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447794

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Z

__inference_<lambda>_3492*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
?
F__inference_dense_12_layer_call_and_return_conditional_losses_24447730

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24446917

inputs#
dense_12_24446911:4

dense_12_24446913:

identity?? dense_12/StatefulPartitionedCall?
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24446898?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_24446911dense_12_24446913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24446910x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
-
+__inference_function_with_signature_1949185?
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
__inference_<lambda>_3492*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
?
F__inference_dense_12_layer_call_and_return_conditional_losses_24446910

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
K
/__inference_sequential_6_layer_call_fn_24447423

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446863`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
/__inference_sequential_4_layer_call_fn_24447372

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446709p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446807
lambda_4_input+
conv2d_2_24446799:
conv2d_2_24446801:
identity?? conv2d_2/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24446675?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_24446799conv2d_2_24446801*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24446687?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24446698?
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24446706r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_4_input
?

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447642

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_24446952

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
value	B :4?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447104

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446819
lambda_4_input+
conv2d_2_24446811:
conv2d_2_24446813:
identity?? conv2d_2/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24446753?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_24446811conv2d_2_24446813*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24446687?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24446698?
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24446706r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_4_input
?
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446863

inputs
identity?
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24446848i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
U
%__inference_get_initial_state_1949903

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
n
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446881
lambda_6_input
identity?
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24446848i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:W S
'
_output_shapes
:?????????2
(
_user_specified_namelambda_6_input
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24446976

inputs#
dense_12_24446970:4

dense_12_24446972:

identity?? dense_12/StatefulPartitionedCall?
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24446952?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_24446970dense_12_24446972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24446910x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_24446706

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_5_layer_call_fn_24446992
lambda_5_input
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24446976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
_user_specified_namelambda_5_input
?
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447184

inputs
identity?
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447181i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447303

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
value	B :4?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????4"
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447397

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????W
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????j
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447616

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447754

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447363
lambda_5_input#
dense_12_24447357:4

dense_12_24447359:

identity?? dense_12/StatefulPartitionedCall?
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447303?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_24447357dense_12_24447359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24447261x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_5_input
?
F
*__inference_re_lu_2_layer_call_fn_24447778

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447049h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447583

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
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
value	B :4?
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   ?
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0?
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447832

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
value	B :4?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?K
?
!__inference__traced_save_24447978
file_prefix*
&savev2_global_step_read_readvariableop	0
,savev2_conv2d_2_kernel_1_read_readvariableop.
*savev2_conv2d_2_bias_1_read_readvariableop0
,savev2_dense_12_kernel_1_read_readvariableop.
*savev2_dense_12_bias_1_read_readvariableop?
?savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel_read_readvariableop?
~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias_read_readvariableop?
?savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel_read_readvariableop?
~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias_read_readvariableop|
xsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel_read_readvariableop?
?savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel_read_readvariableopz
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_global_step_read_readvariableop,savev2_conv2d_2_kernel_1_read_readvariableop*savev2_conv2d_2_bias_1_read_readvariableop,savev2_dense_12_kernel_1_read_readvariableop*savev2_dense_12_bias_1_read_readvariableop?savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel_read_readvariableop~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias_read_readvariableop?savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel_read_readvariableop~savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias_read_readvariableopxsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel_read_readvariableop?savev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel_read_readvariableopvsavev2_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_bias_read_readvariableoposavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableopmsavev2_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableophsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernel_read_readvariableopfsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_bias_read_readvariableophsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernel_read_readvariableopfsavev2_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_bias_read_readvariableop`savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernel_read_readvariableopjsavev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernel_read_readvariableop^savev2_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_bias_read_readvariableopHsavev2_adversary_env_valuernnnetwork_dense_17_kernel_read_readvariableopFsavev2_adversary_env_valuernnnetwork_dense_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?: : :::4
:
:	? : :  : :	 ?:
??:?:
??:?:::4
:
:	? : :  : :	 ?:	(?:?:(:: 2(
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
:	? : 
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
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:,(
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
:	? : 
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447002
lambda_5_input#
dense_12_24446996:4

dense_12_24446998:

identity?? dense_12/StatefulPartitionedCall?
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24446898?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_24446996dense_12_24446998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24446910x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_5_input
?
?
/__inference_sequential_5_layer_call_fn_24447343
lambda_5_input
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447327o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
_user_specified_namelambda_5_input
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447012
lambda_5_input#
dense_12_24447006:4

dense_12_24447008:

identity?? dense_12/StatefulPartitionedCall?
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24446952?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_24447006dense_12_24447008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24446910x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namelambda_5_input
?
?
/__inference_sequential_4_layer_call_fn_24447146
lambda_4_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447130p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_4_input
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446709

inputs+
conv2d_2_24446688:
conv2d_2_24446690:
identity?? conv2d_2/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24446675?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_24446688conv2d_2_24446690*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24446687?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24446698?
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24446706r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_12_layer_call_fn_24447851

inputs
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24447261o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????4: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447677

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447268

inputs#
dense_12_24447262:4

dense_12_24447264:

identity?? dense_12/StatefulPartitionedCall?
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447249?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_24447262dense_12_24447264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24447261x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_4_layer_call_fn_24447490

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447060p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447465

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
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
value	B :4?
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   ?
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0?
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference_action_1949057
	time_step
time_step_1
time_step_2
time_step_3
time_step_4
time_step_5
policy_state
policy_state_1?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4
?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:
?
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	? ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: ?
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: ?
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 ??
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:
???
tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	?r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
??n
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	?
identity	

identity_1

identity_2??gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp?kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp?jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp?lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp@
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
T0*3
_output_shapes!
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_4QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2?
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimstime_step_5QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:??????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:???????????
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
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
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:????????? ?
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:??????????
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:????????? ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:???????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
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
:?????????Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 R??
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

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????2:?????????:??????????:??????????: : : : : : : : : : : : : 2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2?
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2?
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2?
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
_user_specified_name	time_step:ZV
/
_output_shapes
:?????????
#
_user_specified_name	time_step:RN
'
_output_shapes
:?????????2
#
_user_specified_name	time_step:RN
'
_output_shapes
:?????????
#
_user_specified_name	time_step:VR
(
_output_shapes
:??????????
&
_user_specified_namepolicy_state:VR
(
_output_shapes
:??????????
&
_user_specified_namepolicy_state
?
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447427

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
(
&__inference_signature_wrapper_24446661?
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
+__inference_function_with_signature_1949185*(
_construction_contextkEagerRuntime*
_input_shapes 
?
F
*__inference_re_lu_2_layer_call_fn_24447647

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24446698h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447431

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
K
/__inference_sequential_6_layer_call_fn_24447418

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446833`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
F__inference_dense_12_layer_call_and_return_conditional_losses_24447261

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_24446753

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447057

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447060

inputs+
conv2d_2_24447039:
conv2d_2_24447041:
identity?? conv2d_2/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447026?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_24447039conv2d_2_24447041*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447038?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447049?
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447057r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447663

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?{
?
$__inference__traced_restore_24448069
file_prefix&
assignvariableop_global_step:	 >
$assignvariableop_1_conv2d_2_kernel_1:0
"assignvariableop_2_conv2d_2_bias_1:6
$assignvariableop_3_dense_12_kernel_1:4
0
"assignvariableop_4_dense_12_bias_1:
?
xassignvariableop_5_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernel:	? ?
vassignvariableop_6_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_bias: ?
xassignvariableop_7_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernel:  ?
vassignvariableop_8_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_bias: ?
passignvariableop_9_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernel:	 ??
{assignvariableop_10_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernel:
??~
oassignvariableop_11_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_bias:	?|
hassignvariableop_12_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernel:
??u
fassignvariableop_13_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_bias:	?=
#assignvariableop_14_conv2d_2_kernel:/
!assignvariableop_15_conv2d_2_bias:5
#assignvariableop_16_dense_12_kernel:4
/
!assignvariableop_17_dense_12_bias:
t
aassignvariableop_18_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernel:	? m
_assignvariableop_19_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_bias: s
aassignvariableop_20_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernel:  m
_assignvariableop_21_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_bias: l
Yassignvariableop_22_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernel:	 ?v
cassignvariableop_23_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernel:	(?f
Wassignvariableop_24_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_bias:	?S
Aassignvariableop_25_adversary_env_valuernnnetwork_dense_17_kernel:(M
?assignvariableop_26_adversary_env_valuernnnetwork_dense_17_bias:
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
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_2_kernel_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_2_bias_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_dense_12_kernel_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_12_bias_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpxassignvariableop_5_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpvassignvariableop_6_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpxassignvariableop_7_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpvassignvariableop_8_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOppassignvariableop_9_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp{assignvariableop_10_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpoassignvariableop_11_adversary_env_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOphassignvariableop_12_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpfassignvariableop_13_adversary_env_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_12_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_12_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpaassignvariableop_18_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp_assignvariableop_19_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_15_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpaassignvariableop_20_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp_assignvariableop_21_adversary_env_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_16_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpYassignvariableop_22_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpcassignvariableop_23_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_recurrent_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpWassignvariableop_24_adversary_env_valuernnnetwork_valuernnnetwork_dynamic_unroll_5_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpAassignvariableop_25_adversary_env_valuernnnetwork_dense_17_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp?assignvariableop_26_adversary_env_valuernnnetwork_dense_17_biasIdentity_26:output:0"/device:CPU:0*
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
?
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447026

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447545

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
S
/__inference_sequential_6_layer_call_fn_24446871
lambda_6_input
identity?
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446863`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:W S
'
_output_shapes
:?????????2
(
_user_specified_namelambda_6_input
?
`
__inference_<lambda>_3489!
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
?
?
&__inference_signature_wrapper_24446640
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
	unknown_3:	? 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 ?
	unknown_8:
??
	unknown_9:	?

unknown_10:
??

unknown_11:	?
identity	

identity_1

identity_2??StatefulPartitionedCall?
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
7:?????????:??????????:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_function_with_signature_1949090k
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
?:?????????:?????????:?????????2:?????????:?????????:?????????:??????????:??????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:d`
/
_output_shapes
:?????????
-
_user_specified_name0/observation/image:_[
'
_output_shapes
:?????????2
0
_user_specified_name0/observation/random_z:`\
'
_output_shapes
:?????????
1
_user_specified_name0/observation/time_step:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:a]
(
_output_shapes
:??????????
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:??????????
1
_user_specified_name1/actor_network_state/1
?
G
+__inference_lambda_5_layer_call_fn_24447822

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
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447303`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447747

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447812

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447623

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447158
lambda_4_input+
conv2d_2_24447150:
conv2d_2_24447152:
identity?? conv2d_2/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447026?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_24447150conv2d_2_24447152*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447038?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447049?
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447057r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_4_input
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447842

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
value	B :4?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????4"
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
+__inference_conv2d_2_layer_call_fn_24447763

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447038w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447130

inputs+
conv2d_2_24447122:
conv2d_2_24447124:
identity?? conv2d_2/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447104?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_24447122conv2d_2_24447124*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447038?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447049?
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447057r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446779

inputs+
conv2d_2_24446771:
conv2d_2_24446773:
identity?? conv2d_2/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24446753?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_24446771conv2d_2_24446773*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24446687?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24446698?
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24446706r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_sequential_6_layer_call_fn_24447541

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447214`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447773

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447652

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_24446830

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
/__inference_sequential_5_layer_call_fn_24447440

inputs
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24446917o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447599

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
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
value	B :4?
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   ?
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0?
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
S
/__inference_sequential_6_layer_call_fn_24446836
lambda_6_input
identity?
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446833`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:W S
'
_output_shapes
:?????????2
(
_user_specified_namelambda_6_input
?
?
/__inference_sequential_5_layer_call_fn_24447275
lambda_5_input
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447268o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
_user_specified_namelambda_5_input
?
G
+__inference_lambda_6_layer_call_fn_24447673

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24446848`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447531

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????W
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????j
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_4_layer_call_fn_24446795
lambda_4_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446779p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_4_input
?

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447038

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
U
%__inference_get_initial_state_1949153

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
?
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24446698

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference_action_1949671
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_image"
time_step_observation_random_z#
time_step_observation_time_step&
"policy_state_actor_network_state_0&
"policy_state_actor_network_state_1?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4
?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:
?
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	? ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: ?
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: ?
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 ??
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:
???
tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	?r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
??n
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	?
identity	

identity_1

identity_2??gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp?kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp?jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp?lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpG
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
ExpandDimstime_step_observation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_observation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2?
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimstime_step_observation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDimstime_step_step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:??????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:???????????
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
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
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:????????? ?
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:??????????
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:????????? ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:???????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
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
:?????????Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 R??
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

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????2:?????????:??????????:??????????: : : : : : : : : : : : : 2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2?
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2?
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2?
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
_user_specified_nametime_step/discount:lh
/
_output_shapes
:?????????
5
_user_specified_nametime_step/observation/image:gc
'
_output_shapes
:?????????2
8
_user_specified_name time_step/observation/random_z:hd
'
_output_shapes
:?????????
9
_user_specified_name!time_step/observation/time_step:lh
(
_output_shapes
:??????????
<
_user_specified_name$"policy_state/actor_network_state/0:lh
(
_output_shapes
:??????????
<
_user_specified_name$"policy_state/actor_network_state/1
?
?
/__inference_sequential_4_layer_call_fn_24447067
lambda_4_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447060p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_4_input
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447701

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
value	B :4?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446876
lambda_6_input
identity?
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24446830i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:W S
'
_output_shapes
:?????????2
(
_user_specified_namelambda_6_input
?	
?
F__inference_dense_12_layer_call_and_return_conditional_losses_24447861

inputs0
matmul_readvariableop_resource:4
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:4
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_24446675

inputs
identity]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aj
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:?????????[
IdentityIdentitytruediv:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_5_layer_call_fn_24447691

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
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24446952`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447049

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_6_layer_call_fn_24447657

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24446706a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_4_layer_call_fn_24447609

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24446753h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_2_layer_call_fn_24447632

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24446687w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447808

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447515

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????W
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????j
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_6_layer_call_fn_24447788

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447057a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_6_layer_call_fn_24447668

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24446830`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
G
+__inference_lambda_4_layer_call_fn_24447740

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447104h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_6_layer_call_fn_24447799

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447181`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447181

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
G
+__inference_lambda_6_layer_call_fn_24447804

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447199`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447681

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
G
+__inference_lambda_5_layer_call_fn_24447817

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
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447249`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????4"
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447413

inputsA
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOpf
lambda_4/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:?????????W
lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
lambda_4/truedivRealDivlambda_4/Cast:y:0lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:??????????
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2Dlambda_4/truediv:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
flatten_6/ReshapeReshapere_lu_2/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????j
IdentityIdentityflatten_6/Reshape:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference_action_1949430
	step_type

reward
discount
observation_image
observation_random_z
observation_time_step
actor_network_state_0
actor_network_state_1?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4
?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:
?
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	? ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: ?
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: ?
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 ??
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:
???
tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	?r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
??n
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	?
identity	

identity_1

identity_2??gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp?kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp?jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp?lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
ExpandDimsobservation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2?
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimsobservation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:??????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:???????????
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
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
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:????????? ?
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:??????????
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:????????? ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:???????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
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
:?????????Z
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 R??
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

Identity_1Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_2Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????2:?????????:??????????:??????????: : : : : : : : : : : : : 2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2?
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2?
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2?
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
discount:b^
/
_output_shapes
:?????????
+
_user_specified_nameobservation/image:]Y
'
_output_shapes
:?????????2
.
_user_specified_nameobservation/random_z:^Z
'
_output_shapes
:?????????
/
_user_specified_nameobservation/time_step:_[
(
_output_shapes
:??????????
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:??????????
/
_user_specified_nameactor_network_state/1
?
?
/__inference_sequential_5_layer_call_fn_24446924
lambda_5_input
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24446917o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
_user_specified_namelambda_5_input
?
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447549

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447249

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
value	B :4?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????4"
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
/__inference_sequential_5_layer_call_fn_24447558

inputs
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447268o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447481

inputs9
'dense_12_matmul_readvariableop_resource:4
6
(dense_12_biasadd_readvariableop_resource:

identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp^
lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
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
value	B :4?
lambda_5/one_hotOneHotinputslambda_5/one_hot/depth:output:0"lambda_5/one_hot/on_value:output:0#lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4g
lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   ?
lambda_5/ReshapeReshapelambda_5/one_hot:output:0lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0?
dense_12/MatMulMatMullambda_5/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
h
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447170
lambda_4_input+
conv2d_2_24447162:
conv2d_2_24447164:
identity?? conv2d_2/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCalllambda_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447104?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0conv2d_2_24447162conv2d_2_24447164*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447038?
re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447049?
flatten_6/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447057r
IdentityIdentity"flatten_6/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i
NoOpNoOp!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_4_input
?
K
/__inference_sequential_6_layer_call_fn_24447536

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
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447184`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_24446898

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
value	B :4?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
k
+__inference_function_with_signature_1949174
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
І
?
#__inference_distribution_fn_1949887
	step_type

reward
discount
observation_image
observation_random_z
observation_time_step
actor_network_state_0
actor_network_state_1?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource:?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource:?
|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource:4
?
}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource:
?
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource:	? ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource: ?
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource:  ~
pactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource: ?
sactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource:	 ??
uactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource:
???
tactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource:	?r
^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource:
??n
_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource:	?
identity	

identity_1	

identity_2	

identity_3

identity_4??gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp?fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp?tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp?sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp?kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp?jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp?lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp?VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp?UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp=
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
ExpandDimsobservation_imageOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimsobservation_random_zQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2?
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2
ExpandDimsobservation_time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:??????????
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3/dim:output:0*
T0*'
_output_shapes
:??????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	?
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ShapeShapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0*
T0*
_output_shapes
:*
out_type0	?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/ReshapeReshapeMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_2:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/CastCastfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*/
_output_shapes
:??????????
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truedivRealDivfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/Cast:y:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:??????????
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2DConv2DiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/lambda_4/truediv:z:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_4_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAddBiasAddmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D:output:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/ReluRelunActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/ReshapeReshapeoActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/re_lu_2/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Const:output:0*
T0*(
_output_shapes
:???????????
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :4?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hotOneHothActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Reshape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/depth:output:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/on_value:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   ?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/ReshapeReshapenActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/one_hot:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp|actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:4
*
dtype0?
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMulMatMulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/lambda_5/Reshape:output:0{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp}actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAddBiasAddnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul:product:0|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concatConcatV2oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/flatten_6/Reshape:output:0hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd:output:0jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????
  ?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/ReshapeReshapeeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/concatenate_2/concat:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMulMatMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten_7/Reshape:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMulMatMulcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/Relu:activations:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAddBiasAddaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul:product:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/ReluReluaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd:output:0*
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
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten_2/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0*
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
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:????????? ?
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_3:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:??????????
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/RankConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/startConst*
_output_shapes
: *
dtype0*
value	B :?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/rangeRange]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/start:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Rank:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range/delta:output:0*
_output_shapes
:?
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concatConcatV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/values_0:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/range:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat/axis:output:0*
N*
T0*
_output_shapes
:?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/concat:output:0*
T0*+
_output_shapes
:????????? ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ShapeShapeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*
_output_shapes
:?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_sliceStridedSliceWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Shape:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_1:output:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1/perm:output:0*
T0
*'
_output_shapes
:??????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:?
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zerosFill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros/Const:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packedPack_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/strided_slice:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:?
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1Fill`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/packed:output:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:???????????
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SqueezeSqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose:y:0*
T0*'
_output_shapes
:????????? *
squeeze_dims
 ?
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1SqueezeXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/transpose_1:y:0*
T0
*#
_output_shapes
:?????????*
squeeze_dims
 ?
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/SelectSelect[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:???????????
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1Select[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze_1:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/zeros_1:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:???????????
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpsactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype0?
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMulMatMulYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Squeeze:output:0rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpuactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/addAddV2eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul:product:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOptactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAddBiasAdd\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add:z:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/splitSplitmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split/split_dim:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/SigmoidSigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:1*
T0*(
_output_shapes
:???????????
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mulMulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_1:y:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/Select_1:output:0*
T0*(
_output_shapes
:???????????
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/TanhTanhcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:2*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1Mul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid:y:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh:y:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1AddV2\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_1:z:0*
T0*(
_output_shapes
:???????????
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2SigmoidcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/split:output:3*
T0*(
_output_shapes
:???????????
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1Tanh^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0*
T0*(
_output_shapes
:???????????
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2MulbActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Sigmoid_2:y:0_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims
ExpandDims^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:???????????
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueeze\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/ExpandDims:output:0*
T0*(
_output_shapes
:??????????*
squeeze_dims
?
UActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0]ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
VActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
GActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddPActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
FActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
@ActorDistributionRnnNetwork/CategoricalProjectionNetwork/ReshapeReshapePActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0OActorDistributionRnnNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
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

Identity_3Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/mul_2:z:0^NoOp*
T0*(
_output_shapes
:???????????

Identity_4Identity^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOpt^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpl^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpk^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpm^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOpW^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpV^ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp*"
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
?:?????????:?????????:?????????:?????????:?????????2:?????????:??????????:??????????: : : : : : : : : : : : : 2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2?
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/BiasAdd/ReadVariableOp2?
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/MatMul/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_4/conv2d_2/Conv2D/ReadVariableOp2?
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOptActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/BiasAdd/ReadVariableOp2?
sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOpsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/sequential_5/dense_12/MatMul/ReadVariableOp2?
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/BiasAdd/ReadVariableOp2?
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul/ReadVariableOp2?
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOplActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/lstm_cell_4/MatMul_1/ReadVariableOp2?
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
discount:b^
/
_output_shapes
:?????????
+
_user_specified_nameobservation/image:]Y
'
_output_shapes
:?????????2
.
_user_specified_nameobservation/random_z:^Z
'
_output_shapes
:?????????
/
_user_specified_nameobservation/time_step:_[
(
_output_shapes
:??????????
/
_user_specified_nameactor_network_state/0:_[
(
_output_shapes
:??????????
/
_user_specified_nameactor_network_state/1
?
?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447327

inputs#
dense_12_24447321:4

dense_12_24447323:

identity?? dense_12/StatefulPartitionedCall?
lambda_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447303?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0dense_12_24447321dense_12_24447323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24447261x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
i
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_4_layer_call_fn_24447499

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447130p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_24446848

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
G
+__inference_lambda_5_layer_call_fn_24447686

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
:?????????4* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_24446898`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_function_with_signature_1949090
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
	unknown_3:	? 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:	 ?
	unknown_8:
??
	unknown_9:	?

unknown_10:
??

unknown_11:	?
identity	

identity_1

identity_2??StatefulPartitionedCall?
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
7:?????????:??????????:??????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_action_1949057k
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
?:?????????:?????????:?????????:?????????:?????????2:?????????:??????????:??????????: : : : : : : : : : : : : 22
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
0/discount:d`
/
_output_shapes
:?????????
-
_user_specified_name0/observation/image:_[
'
_output_shapes
:?????????2
0
_user_specified_name0/observation/random_z:`\
'
_output_shapes
:?????????
1
_user_specified_name0/observation/time_step:a]
(
_output_shapes
:??????????
1
_user_specified_name1/actor_network_state/0:a]
(
_output_shapes
:??????????
1
_user_specified_name1/actor_network_state/1
?
V
&__inference_signature_wrapper_24446649

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
+__inference_function_with_signature_1949158a
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
?
S
/__inference_sequential_6_layer_call_fn_24447187
lambda_6_input
identity?
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447184`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:W S
'
_output_shapes
:?????????2
(
_user_specified_namelambda_6_input
?
G
+__inference_lambda_4_layer_call_fn_24447735

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447026h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
S
/__inference_sequential_6_layer_call_fn_24447222
lambda_6_input
identity?
PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447214`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:W S
'
_output_shapes
:?????????2
(
_user_specified_namelambda_6_input
?

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24446687

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
[
+__inference_function_with_signature_1949158

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
%__inference_get_initial_state_1949153a
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
?
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446833

inputs
identity?
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24446830i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447711

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
value	B :4?
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????4^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????4   n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????4X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????4"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447199

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
/__inference_sequential_4_layer_call_fn_24446716
lambda_4_input!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446709p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_namelambda_4_input
?
?
/__inference_sequential_5_layer_call_fn_24447567

inputs
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447327o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
f
&__inference_signature_wrapper_24446657
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
+__inference_function_with_signature_1949174^
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
+__inference_dense_12_layer_call_fn_24447720

inputs
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_24446910o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????4: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????4
 
_user_specified_nameinputs
?
?
/__inference_sequential_5_layer_call_fn_24447449

inputs
unknown:4

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_5_layer_call_and_return_conditional_losses_24446976o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
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
n
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447232
lambda_6_input
identity?
lambda_6/PartitionedCallPartitionedCalllambda_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447199i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:W S
'
_output_shapes
:?????????2
(
_user_specified_namelambda_6_input
?
f
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447214

inputs
identity?
lambda_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447199i
IdentityIdentity!lambda_6/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0_discount:0?????????
R
0/observation/image;
action_0_observation_image:0?????????
P
0/observation/random_z6
action_0_observation_random_z:0?????????2
R
0/observation/time_step7
 action_0_observation_time_step:0?????????
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
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:??
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
__inference_action_1949430
__inference_action_1949671?
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
#__inference_distribution_fn_1949887?
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
%__inference_get_initial_state_1949903?
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
__inference_<lambda>_3492"?
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
__inference_<lambda>_3489"?
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
):'2conv2d_2/kernel
:2conv2d_2/bias
!:4
2dense_12/kernel
:
2dense_12/bias
x:v	? 2eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/kernel
q:o 2cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_13/bias
w:u  2eadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/kernel
q:o 2cadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_14/bias
p:n	 ?2]adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/kernel
{:y
??2gadversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/recurrent_kernel
j:h?2[adversary_env/ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll_4/bias
h:f
??2Tadversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/kernel
a:_?2Radversary_env/ActorDistributionRnnNetwork/CategoricalProjectionNetwork/logits/bias
):'2conv2d_2/kernel
:2conv2d_2/bias
!:4
2dense_12/kernel
:
2dense_12/bias
`:^	? 2Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/kernel
Y:W 2Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_15/bias
_:]  2Madversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/kernel
Y:W 2Kadversary_env/ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_16/bias
X:V	 ?2Eadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/kernel
b:`	(?2Oadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/recurrent_kernel
R:P?2Cadversary_env/ValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_5/bias
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
?
6_actor_network
7_time_step_spec
8_policy_state_spec
9_policy_step_spec
:_trajectory_spec
;_value_network"
_generic_user_object
?B?
__inference_action_1949430	step_typerewarddiscountobservation/imageobservation/random_zobservation/time_stepactor_network_state/0actor_network_state/1"?
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
__inference_action_1949671time_step/step_typetime_step/rewardtime_step/discounttime_step/observation/imagetime_step/observation/random_ztime_step/observation/time_step"policy_state/actor_network_state/0"policy_state/actor_network_state/1"?
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
#__inference_distribution_fn_1949887	step_typerewarddiscountobservation/imageobservation/random_zobservation/time_stepactor_network_state/0actor_network_state/1"?
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
%__inference_get_initial_state_1949903
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
&__inference_signature_wrapper_24446640
0/discount0/observation/image0/observation/random_z0/observation/time_step0/reward0/step_type1/actor_network_state/01/actor_network_state/1"?
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
&__inference_signature_wrapper_24446649
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
&__inference_signature_wrapper_24446657"?
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
&__inference_signature_wrapper_24446661"?
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
8
?0
?1
?2"
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
8
?0
?1
?2"
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
X
?0
?1
?2
?3
?4
?5
?6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?layer-0
?layer_with_weights-0
?layer-1
?layer-2
?layer-3
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
?layer-0
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
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
X
?0
?1
?2
?3
?4
?5
?6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
?
?layer-0
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
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
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
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
+?&call_and_return_all_conditional_losses"
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
/__inference_sequential_4_layer_call_fn_24446716
/__inference_sequential_4_layer_call_fn_24447372
/__inference_sequential_4_layer_call_fn_24447381
/__inference_sequential_4_layer_call_fn_24446795?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447397
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447413
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446807
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446819?
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
?
?trace_0
?trace_1
?trace_2
?trace_32?
/__inference_sequential_6_layer_call_fn_24446836
/__inference_sequential_6_layer_call_fn_24447418
/__inference_sequential_6_layer_call_fn_24447423
/__inference_sequential_6_layer_call_fn_24446871?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447427
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447431
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446876
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446881?
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
/__inference_sequential_5_layer_call_fn_24446924
/__inference_sequential_5_layer_call_fn_24447440
/__inference_sequential_5_layer_call_fn_24447449
/__inference_sequential_5_layer_call_fn_24446992?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447465
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447481
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447002
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447012?
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
bias
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
/__inference_sequential_4_layer_call_fn_24447067
/__inference_sequential_4_layer_call_fn_24447490
/__inference_sequential_4_layer_call_fn_24447499
/__inference_sequential_4_layer_call_fn_24447146?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447515
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447531
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447158
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447170?
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
?
?trace_0
?trace_1
?trace_2
?trace_32?
/__inference_sequential_6_layer_call_fn_24447187
/__inference_sequential_6_layer_call_fn_24447536
/__inference_sequential_6_layer_call_fn_24447541
/__inference_sequential_6_layer_call_fn_24447222?
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
?trace_2
?trace_32?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447545
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447549
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447227
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447232?
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
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

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
?
?trace_0
?trace_1
?trace_2
?trace_32?
/__inference_sequential_5_layer_call_fn_24447275
/__inference_sequential_5_layer_call_fn_24447558
/__inference_sequential_5_layer_call_fn_24447567
/__inference_sequential_5_layer_call_fn_24447343?
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
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447583
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447599
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447353
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447363?
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
 z?trace_0z?trace_1z?trace_2z?trace_3
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
+__inference_lambda_4_layer_call_fn_24447604
+__inference_lambda_4_layer_call_fn_24447609?
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
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447616
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447623?
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
+__inference_conv2d_2_layer_call_fn_24447632?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447642?
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
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_re_lu_2_layer_call_fn_24447647?
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
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447652?
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
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_flatten_6_layer_call_fn_24447657?
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447663?
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
?0
?1
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
/__inference_sequential_4_layer_call_fn_24446716lambda_4_input"?
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
/__inference_sequential_4_layer_call_fn_24447372inputs"?
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
/__inference_sequential_4_layer_call_fn_24447381inputs"?
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
/__inference_sequential_4_layer_call_fn_24446795lambda_4_input"?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447397inputs"?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447413inputs"?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446807lambda_4_input"?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446819lambda_4_input"?
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
+__inference_lambda_6_layer_call_fn_24447668
+__inference_lambda_6_layer_call_fn_24447673?
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
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447677
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447681?
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
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_sequential_6_layer_call_fn_24446836lambda_6_input"?
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
/__inference_sequential_6_layer_call_fn_24447418inputs"?
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
/__inference_sequential_6_layer_call_fn_24447423inputs"?
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
/__inference_sequential_6_layer_call_fn_24446871lambda_6_input"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447427inputs"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447431inputs"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446876lambda_6_input"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446881lambda_6_input"?
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
+__inference_lambda_5_layer_call_fn_24447686
+__inference_lambda_5_layer_call_fn_24447691?
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
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447701
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447711?
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
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_dense_12_layer_call_fn_24447720?
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
F__inference_dense_12_layer_call_and_return_conditional_losses_24447730?
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
/__inference_sequential_5_layer_call_fn_24446924lambda_5_input"?
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
/__inference_sequential_5_layer_call_fn_24447440inputs"?
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
/__inference_sequential_5_layer_call_fn_24447449inputs"?
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
/__inference_sequential_5_layer_call_fn_24446992lambda_5_input"?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447465inputs"?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447481inputs"?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447002lambda_5_input"?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447012lambda_5_input"?
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
+__inference_lambda_4_layer_call_fn_24447735
+__inference_lambda_4_layer_call_fn_24447740?
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
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447747
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447754?
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
+__inference_conv2d_2_layer_call_fn_24447763?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447773?
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
*__inference_re_lu_2_layer_call_fn_24447778?
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
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447783?
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
,__inference_flatten_6_layer_call_fn_24447788?
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447794?
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
/__inference_sequential_4_layer_call_fn_24447067lambda_4_input"?
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
/__inference_sequential_4_layer_call_fn_24447490inputs"?
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
/__inference_sequential_4_layer_call_fn_24447499inputs"?
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
/__inference_sequential_4_layer_call_fn_24447146lambda_4_input"?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447515inputs"?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447531inputs"?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447158lambda_4_input"?
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447170lambda_4_input"?
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
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_lambda_6_layer_call_fn_24447799
+__inference_lambda_6_layer_call_fn_24447804?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447808
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447812?
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
 z?trace_0z?trace_1
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_sequential_6_layer_call_fn_24447187lambda_6_input"?
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
/__inference_sequential_6_layer_call_fn_24447536inputs"?
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
/__inference_sequential_6_layer_call_fn_24447541inputs"?
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
/__inference_sequential_6_layer_call_fn_24447222lambda_6_input"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447545inputs"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447549inputs"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447227lambda_6_input"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447232lambda_6_input"?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_lambda_5_layer_call_fn_24447817
+__inference_lambda_5_layer_call_fn_24447822?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447832
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447842?
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
 z?trace_0z?trace_1
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_dense_12_layer_call_fn_24447851?
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
 z?trace_0
?
?trace_02?
F__inference_dense_12_layer_call_and_return_conditional_losses_24447861?
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
 z?trace_0
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
/__inference_sequential_5_layer_call_fn_24447275lambda_5_input"?
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
/__inference_sequential_5_layer_call_fn_24447558inputs"?
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
/__inference_sequential_5_layer_call_fn_24447567inputs"?
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
/__inference_sequential_5_layer_call_fn_24447343lambda_5_input"?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447583inputs"?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447599inputs"?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447353lambda_5_input"?
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447363lambda_5_input"?
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
+__inference_lambda_4_layer_call_fn_24447604inputs"?
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
+__inference_lambda_4_layer_call_fn_24447609inputs"?
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
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447616inputs"?
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
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447623inputs"?
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
+__inference_conv2d_2_layer_call_fn_24447632inputs"?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447642inputs"?
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
*__inference_re_lu_2_layer_call_fn_24447647inputs"?
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
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447652inputs"?
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
,__inference_flatten_6_layer_call_fn_24447657inputs"?
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447663inputs"?
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
+__inference_lambda_6_layer_call_fn_24447668inputs"?
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
+__inference_lambda_6_layer_call_fn_24447673inputs"?
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
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447677inputs"?
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
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447681inputs"?
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
+__inference_lambda_5_layer_call_fn_24447686inputs"?
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
+__inference_lambda_5_layer_call_fn_24447691inputs"?
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
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447701inputs"?
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
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447711inputs"?
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
+__inference_dense_12_layer_call_fn_24447720inputs"?
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
F__inference_dense_12_layer_call_and_return_conditional_losses_24447730inputs"?
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
+__inference_lambda_4_layer_call_fn_24447735inputs"?
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
+__inference_lambda_4_layer_call_fn_24447740inputs"?
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
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447747inputs"?
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
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447754inputs"?
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
+__inference_conv2d_2_layer_call_fn_24447763inputs"?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447773inputs"?
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
*__inference_re_lu_2_layer_call_fn_24447778inputs"?
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
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447783inputs"?
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
,__inference_flatten_6_layer_call_fn_24447788inputs"?
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447794inputs"?
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
+__inference_lambda_6_layer_call_fn_24447799inputs"?
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
+__inference_lambda_6_layer_call_fn_24447804inputs"?
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
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447808inputs"?
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
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447812inputs"?
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
+__inference_lambda_5_layer_call_fn_24447817inputs"?
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
+__inference_lambda_5_layer_call_fn_24447822inputs"?
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
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447832inputs"?
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
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447842inputs"?
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
+__inference_dense_12_layer_call_fn_24447851inputs"?
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
F__inference_dense_12_layer_call_and_return_conditional_losses_24447861inputs"?
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
__inference_<lambda>_3489?

? 
? "? 	1
__inference_<lambda>_3492?

? 
? "? ?
__inference_action_1949430????
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount??????????
observation???
<
image3?0
observation/image?????????
:
random_z.?+
observation/random_z?????????2
<
	time_step/?,
observation/time_step?????????
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
__inference_action_1949671????
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount??????????
observation???
F
image=?:
time_step/observation/image?????????
D
random_z8?5
time_step/observation/random_z?????????2
F
	time_step9?6
time_step/observation/time_step?????????
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447642l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_24447773l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_2_layer_call_fn_24447632_7?4
-?*
(?%
inputs?????????
? " ???????????
+__inference_conv2d_2_layer_call_fn_24447763_7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_dense_12_layer_call_and_return_conditional_losses_24447730\/?,
%?"
 ?
inputs?????????4
? "%?"
?
0?????????

? ?
F__inference_dense_12_layer_call_and_return_conditional_losses_24447861\/?,
%?"
 ?
inputs?????????4
? "%?"
?
0?????????

? ~
+__inference_dense_12_layer_call_fn_24447720O/?,
%?"
 ?
inputs?????????4
? "??????????
~
+__inference_dense_12_layer_call_fn_24447851O/?,
%?"
 ?
inputs?????????4
? "??????????
?
#__inference_distribution_fn_1949887????
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount??????????
observation???
<
image3?0
observation/image?????????
:
random_z.?+
observation/random_z?????????2
<
	time_step/?,
observation/time_step?????????
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447663a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
G__inference_flatten_6_layer_call_and_return_conditional_losses_24447794a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
,__inference_flatten_6_layer_call_fn_24447657T7?4
-?*
(?%
inputs?????????
? "????????????
,__inference_flatten_6_layer_call_fn_24447788T7?4
-?*
(?%
inputs?????????
? "????????????
%__inference_get_initial_state_1949903?"?
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
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447616p??<
5?2
(?%
inputs?????????

 
p 
? "-?*
#? 
0?????????
? ?
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447623p??<
5?2
(?%
inputs?????????

 
p
? "-?*
#? 
0?????????
? ?
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447747p??<
5?2
(?%
inputs?????????

 
p 
? "-?*
#? 
0?????????
? ?
F__inference_lambda_4_layer_call_and_return_conditional_losses_24447754p??<
5?2
(?%
inputs?????????

 
p
? "-?*
#? 
0?????????
? ?
+__inference_lambda_4_layer_call_fn_24447604c??<
5?2
(?%
inputs?????????

 
p 
? " ???????????
+__inference_lambda_4_layer_call_fn_24447609c??<
5?2
(?%
inputs?????????

 
p
? " ???????????
+__inference_lambda_4_layer_call_fn_24447735c??<
5?2
(?%
inputs?????????

 
p 
? " ???????????
+__inference_lambda_4_layer_call_fn_24447740c??<
5?2
(?%
inputs?????????

 
p
? " ???????????
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447701`7?4
-?*
 ?
inputs?????????

 
p 
? "%?"
?
0?????????4
? ?
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447711`7?4
-?*
 ?
inputs?????????

 
p
? "%?"
?
0?????????4
? ?
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447832`7?4
-?*
 ?
inputs?????????

 
p 
? "%?"
?
0?????????4
? ?
F__inference_lambda_5_layer_call_and_return_conditional_losses_24447842`7?4
-?*
 ?
inputs?????????

 
p
? "%?"
?
0?????????4
? ?
+__inference_lambda_5_layer_call_fn_24447686S7?4
-?*
 ?
inputs?????????

 
p 
? "??????????4?
+__inference_lambda_5_layer_call_fn_24447691S7?4
-?*
 ?
inputs?????????

 
p
? "??????????4?
+__inference_lambda_5_layer_call_fn_24447817S7?4
-?*
 ?
inputs?????????

 
p 
? "??????????4?
+__inference_lambda_5_layer_call_fn_24447822S7?4
-?*
 ?
inputs?????????

 
p
? "??????????4?
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447677`7?4
-?*
 ?
inputs?????????2

 
p 
? "%?"
?
0?????????2
? ?
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447681`7?4
-?*
 ?
inputs?????????2

 
p
? "%?"
?
0?????????2
? ?
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447808`7?4
-?*
 ?
inputs?????????2

 
p 
? "%?"
?
0?????????2
? ?
F__inference_lambda_6_layer_call_and_return_conditional_losses_24447812`7?4
-?*
 ?
inputs?????????2

 
p
? "%?"
?
0?????????2
? ?
+__inference_lambda_6_layer_call_fn_24447668S7?4
-?*
 ?
inputs?????????2

 
p 
? "??????????2?
+__inference_lambda_6_layer_call_fn_24447673S7?4
-?*
 ?
inputs?????????2

 
p
? "??????????2?
+__inference_lambda_6_layer_call_fn_24447799S7?4
-?*
 ?
inputs?????????2

 
p 
? "??????????2?
+__inference_lambda_6_layer_call_fn_24447804S7?4
-?*
 ?
inputs?????????2

 
p
? "??????????2?
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447652h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
E__inference_re_lu_2_layer_call_and_return_conditional_losses_24447783h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_re_lu_2_layer_call_fn_24447647[7?4
-?*
(?%
inputs?????????
? " ???????????
*__inference_re_lu_2_layer_call_fn_24447778[7?4
-?*
(?%
inputs?????????
? " ???????????
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446807uG?D
=?:
0?-
lambda_4_input?????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24446819uG?D
=?:
0?-
lambda_4_input?????????
p

 
? "&?#
?
0??????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447158uG?D
=?:
0?-
lambda_4_input?????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447170uG?D
=?:
0?-
lambda_4_input?????????
p

 
? "&?#
?
0??????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447397m??<
5?2
(?%
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447413m??<
5?2
(?%
inputs?????????
p

 
? "&?#
?
0??????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447515m??<
5?2
(?%
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_24447531m??<
5?2
(?%
inputs?????????
p

 
? "&?#
?
0??????????
? ?
/__inference_sequential_4_layer_call_fn_24446716hG?D
=?:
0?-
lambda_4_input?????????
p 

 
? "????????????
/__inference_sequential_4_layer_call_fn_24446795hG?D
=?:
0?-
lambda_4_input?????????
p

 
? "????????????
/__inference_sequential_4_layer_call_fn_24447067hG?D
=?:
0?-
lambda_4_input?????????
p 

 
? "????????????
/__inference_sequential_4_layer_call_fn_24447146hG?D
=?:
0?-
lambda_4_input?????????
p

 
? "????????????
/__inference_sequential_4_layer_call_fn_24447372`??<
5?2
(?%
inputs?????????
p 

 
? "????????????
/__inference_sequential_4_layer_call_fn_24447381`??<
5?2
(?%
inputs?????????
p

 
? "????????????
/__inference_sequential_4_layer_call_fn_24447490`??<
5?2
(?%
inputs?????????
p 

 
? "????????????
/__inference_sequential_4_layer_call_fn_24447499`??<
5?2
(?%
inputs?????????
p

 
? "????????????
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447002l??<
5?2
(?%
lambda_5_input?????????
p 

 
? "%?"
?
0?????????

? ?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447012l??<
5?2
(?%
lambda_5_input?????????
p

 
? "%?"
?
0?????????

? ?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447353l??<
5?2
(?%
lambda_5_input?????????
p 

 
? "%?"
?
0?????????

? ?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447363l??<
5?2
(?%
lambda_5_input?????????
p

 
? "%?"
?
0?????????

? ?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447465d7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447481d7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????

? ?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447583d7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
J__inference_sequential_5_layer_call_and_return_conditional_losses_24447599d7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????

? ?
/__inference_sequential_5_layer_call_fn_24446924_??<
5?2
(?%
lambda_5_input?????????
p 

 
? "??????????
?
/__inference_sequential_5_layer_call_fn_24446992_??<
5?2
(?%
lambda_5_input?????????
p

 
? "??????????
?
/__inference_sequential_5_layer_call_fn_24447275_??<
5?2
(?%
lambda_5_input?????????
p 

 
? "??????????
?
/__inference_sequential_5_layer_call_fn_24447343_??<
5?2
(?%
lambda_5_input?????????
p

 
? "??????????
?
/__inference_sequential_5_layer_call_fn_24447440W7?4
-?*
 ?
inputs?????????
p 

 
? "??????????
?
/__inference_sequential_5_layer_call_fn_24447449W7?4
-?*
 ?
inputs?????????
p

 
? "??????????
?
/__inference_sequential_5_layer_call_fn_24447558W7?4
-?*
 ?
inputs?????????
p 

 
? "??????????
?
/__inference_sequential_5_layer_call_fn_24447567W7?4
-?*
 ?
inputs?????????
p

 
? "??????????
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446876h??<
5?2
(?%
lambda_6_input?????????2
p 

 
? "%?"
?
0?????????2
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24446881h??<
5?2
(?%
lambda_6_input?????????2
p

 
? "%?"
?
0?????????2
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447227h??<
5?2
(?%
lambda_6_input?????????2
p 

 
? "%?"
?
0?????????2
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447232h??<
5?2
(?%
lambda_6_input?????????2
p

 
? "%?"
?
0?????????2
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447427`7?4
-?*
 ?
inputs?????????2
p 

 
? "%?"
?
0?????????2
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447431`7?4
-?*
 ?
inputs?????????2
p

 
? "%?"
?
0?????????2
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447545`7?4
-?*
 ?
inputs?????????2
p 

 
? "%?"
?
0?????????2
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_24447549`7?4
-?*
 ?
inputs?????????2
p

 
? "%?"
?
0?????????2
? ?
/__inference_sequential_6_layer_call_fn_24446836[??<
5?2
(?%
lambda_6_input?????????2
p 

 
? "??????????2?
/__inference_sequential_6_layer_call_fn_24446871[??<
5?2
(?%
lambda_6_input?????????2
p

 
? "??????????2?
/__inference_sequential_6_layer_call_fn_24447187[??<
5?2
(?%
lambda_6_input?????????2
p 

 
? "??????????2?
/__inference_sequential_6_layer_call_fn_24447222[??<
5?2
(?%
lambda_6_input?????????2
p

 
? "??????????2?
/__inference_sequential_6_layer_call_fn_24447418S7?4
-?*
 ?
inputs?????????2
p 

 
? "??????????2?
/__inference_sequential_6_layer_call_fn_24447423S7?4
-?*
 ?
inputs?????????2
p

 
? "??????????2?
/__inference_sequential_6_layer_call_fn_24447536S7?4
-?*
 ?
inputs?????????2
p 

 
? "??????????2?
/__inference_sequential_6_layer_call_fn_24447541S7?4
-?*
 ?
inputs?????????2
p

 
? "??????????2?
&__inference_signature_wrapper_24446640????
? 
???
.

0/discount ?

0/discount?????????
L
0/observation/image5?2
0/observation/image?????????
J
0/observation/random_z0?-
0/observation/random_z?????????2
L
0/observation/time_step1?.
0/observation/time_step?????????
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
&__inference_signature_wrapper_24446649?0?-
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
&__inference_signature_wrapper_244466570?

? 
? "?

int64?
int64 	>
&__inference_signature_wrapper_24446661?

? 
? "? 