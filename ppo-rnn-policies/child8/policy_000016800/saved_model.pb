д"
м
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
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
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
@
Softplus
features"T
activations"T"
Ttype:
2
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
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
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
;
Sub
x"T
y"T
z"T"
Ttype:
2	
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
 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8За
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

TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*e
shared_nameVTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel
ў
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	Ш*
dtype0
§
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*c
shared_nameTRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias
і
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOpRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:Ш*
dtype0

VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Шd*g
shared_nameXVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel

jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOpVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel*
_output_shapes
:	Шd*
dtype0

TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*e
shared_nameVTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias
љ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias*
_output_shapes
:d*
dtype0

WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*h
shared_nameYWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernel

kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernel*
_output_shapes
:	d*
dtype0
 
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*r
shared_namecaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernel

uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*f
shared_nameWUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/bias
ќ
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/bias/Read/ReadVariableOpReadVariableOpUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/bias*
_output_shapes	
:*
dtype0

YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*j
shared_name[YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernel

mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOpYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernel* 
_output_shapes
:
*
dtype0
Є
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*t
shared_nameecActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernel

wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernel* 
_output_shapes
:
*
dtype0

WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*h
shared_nameYWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/bias

kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOpWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/bias*
_output_shapes	
:*
dtype0
о
CActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias
з
WActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpReadVariableOpCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias*
_output_shapes
:*
dtype0
џ
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*b
shared_nameSQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel
ј
eActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOpReadVariableOpQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel*
_output_shapes
:	*
dtype0
і
OActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*`
shared_nameQOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias
я
cActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpReadVariableOpOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias*
_output_shapes
:*
dtype0
й
>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ш*O
shared_name@>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel
в
RValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel*
_output_shapes
:	Ш*
dtype0
б
<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*M
shared_name><ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias
Ъ
PValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOp<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias*
_output_shapes	
:Ш*
dtype0
й
>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Шd*O
shared_name@>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel
в
RValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOp>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel*
_output_shapes
:	Шd*
dtype0
а
<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*M
shared_name><ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias
Щ
PValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOp<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias*
_output_shapes
:d*
dtype0
у
CValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*T
shared_nameECValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/kernel
м
WValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOpCValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/kernel*
_output_shapes
:	d*
dtype0
ј
MValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*^
shared_nameOMValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/recurrent_kernel
ё
aValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOpMValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/recurrent_kernel* 
_output_shapes
:
*
dtype0
л
AValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/bias
д
UValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOpAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/bias*
_output_shapes	
:*
dtype0
ф
CValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*T
shared_nameECValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/kernel
н
WValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOpCValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/kernel* 
_output_shapes
:
*
dtype0
ј
MValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*^
shared_nameOMValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/recurrent_kernel
ё
aValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOpMValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/recurrent_kernel* 
_output_shapes
:
*
dtype0
л
AValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/bias
д
UValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOpAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/bias*
_output_shapes	
:*
dtype0

ValueRnnNetwork/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name ValueRnnNetwork/dense_4/kernel

2ValueRnnNetwork/dense_4/kernel/Read/ReadVariableOpReadVariableOpValueRnnNetwork/dense_4/kernel*
_output_shapes
:	*
dtype0

ValueRnnNetwork/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameValueRnnNetwork/dense_4/bias

0ValueRnnNetwork/dense_4/bias/Read/ReadVariableOpReadVariableOpValueRnnNetwork/dense_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ьj
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*j
value§iBњi Bѓi
k
policy_state_spec

train_step
metadata
model_variables
_all_assets

signatures

actor_network_state
FD
VARIABLE_VALUEglobal_step%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
О
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24

!0
"1
#2
 

$0
%1

VARIABLE_VALUETActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUETActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ЄЁ
VARIABLE_VALUEaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernel,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ІЃ
VARIABLE_VALUEcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/bias,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEMValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/recurrent_kernel-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/bias-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/kernel-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEMValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/recurrent_kernel-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/bias-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEValueRnnNetwork/dense_4/kernel-model_variables/23/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEValueRnnNetwork/dense_4/bias-model_variables/24/.ATTRIBUTES/VARIABLE_VALUE

&ref
&1

'ref
'1

(ref
(1
 
 

actor_network_state

	&state
&1
W
)_actor_network
&_policy_state_spec
*_policy_step_spec
+_value_network

_state_spec
,_lstm_encoder
-_projection_networks
.	variables
/regularization_losses
0trainable_variables
1	keras_api

	&state
&1

2_state_spec
3_lstm_encoder
4_postprocessing_layers
5	variables
6regularization_losses
7trainable_variables
8	keras_api

_state_spec
9_input_encoder
:_lstm_network
;_output_encoder
<	variables
=regularization_losses
>trainable_variables
?	keras_api
z
@_means_projection_layer
	A_bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
^
0
	1

2
3
4
5
6
7
8
9
10
11
12
 
^
0
	1

2
3
4
5
6
7
8
9
10
11
12
­
Fmetrics
.	variables

Glayers
Hnon_trainable_variables
Ilayer_metrics
/regularization_losses
Jlayer_regularization_losses
0trainable_variables

K0
L1

2_state_spec
M_input_encoder
N_lstm_network
O_output_encoder
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
h

kernel
 bias
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
V
0
1
2
3
4
5
6
7
8
9
10
 11
 
V
0
1
2
3
4
5
6
7
8
9
10
 11
­
Xmetrics
5	variables

Ylayers
Znon_trainable_variables
[layer_metrics
6regularization_losses
\layer_regularization_losses
7trainable_variables
n
]_postprocessing_layers
^	variables
_regularization_losses
`trainable_variables
a	keras_api
\
bcell
c	variables
dregularization_losses
etrainable_variables
f	keras_api
 
F
0
	1

2
3
4
5
6
7
8
9
 
F
0
	1

2
3
4
5
6
7
8
9
­
gmetrics
<	variables

hlayers
inon_trainable_variables
jlayer_metrics
=regularization_losses
klayer_regularization_losses
>trainable_variables
h

kernel
bias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
\
bias
p	variables
qregularization_losses
rtrainable_variables
s	keras_api

0
1
2
 

0
1
2
­
tmetrics
B	variables

ulayers
vnon_trainable_variables
wlayer_metrics
Cregularization_losses
xlayer_regularization_losses
Dtrainable_variables
 

,0
-1
 
 
 
 
 
n
y_postprocessing_layers
z	variables
{regularization_losses
|trainable_variables
}	keras_api
_
~cell
	variables
regularization_losses
trainable_variables
	keras_api
 
F
0
1
2
3
4
5
6
7
8
9
 
F
0
1
2
3
4
5
6
7
8
9
В
metrics
P	variables
layers
non_trainable_variables
layer_metrics
Qregularization_losses
 layer_regularization_losses
Rtrainable_variables

0
 1
 

0
 1
В
metrics
T	variables
layers
non_trainable_variables
layer_metrics
Uregularization_losses
 layer_regularization_losses
Vtrainable_variables
 

30
41
 
 
 

0
1
2

0
	1

2
3
 

0
	1

2
3
В
metrics
^	variables
layers
non_trainable_variables
layer_metrics
_regularization_losses
 layer_regularization_losses
`trainable_variables
b

cells
	variables
regularization_losses
trainable_variables
	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
В
metrics
c	variables
layers
non_trainable_variables
layer_metrics
dregularization_losses
 layer_regularization_losses
etrainable_variables
 

90
:1
 
 
 

0
1
 

0
1
В
metrics
l	variables
 layers
Ёnon_trainable_variables
Ђlayer_metrics
mregularization_losses
 Ѓlayer_regularization_losses
ntrainable_variables

0
 

0
В
Єmetrics
p	variables
Ѕlayers
Іnon_trainable_variables
Їlayer_metrics
qregularization_losses
 Јlayer_regularization_losses
rtrainable_variables
 

@0
A1
 
 
 

Љ0
Њ1
Ћ2

0
1
2
3
 

0
1
2
3
В
Ќmetrics
z	variables
­layers
Ўnon_trainable_variables
Џlayer_metrics
{regularization_losses
 Аlayer_regularization_losses
|trainable_variables
b

Бcells
В	variables
Гregularization_losses
Дtrainable_variables
Е	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
Д
Жmetrics
	variables
Зlayers
Иnon_trainable_variables
Йlayer_metrics
regularization_losses
 Кlayer_regularization_losses
trainable_variables
 

M0
N1
 
 
 
 
 
 
 
 
V
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
l

kernel
	bias
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
l


kernel
bias
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
 

0
1
2
 
 
 

Ч0
Ш1
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
Е
Щmetrics
	variables
Ъlayers
Ыnon_trainable_variables
Ьlayer_metrics
regularization_losses
 Эlayer_regularization_losses
trainable_variables
 

b0
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
V
Ю	variables
Яregularization_losses
аtrainable_variables
б	keras_api
l

kernel
bias
в	variables
гregularization_losses
дtrainable_variables
е	keras_api
l

kernel
bias
ж	variables
зregularization_losses
иtrainable_variables
й	keras_api
 

Љ0
Њ1
Ћ2
 
 
 

к0
л1
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
Е
мmetrics
В	variables
нlayers
оnon_trainable_variables
пlayer_metrics
Гregularization_losses
 рlayer_regularization_losses
Дtrainable_variables
 

~0
 
 
 
 
 
 
Е
сmetrics
Л	variables
тlayers
уnon_trainable_variables
фlayer_metrics
Мregularization_losses
 хlayer_regularization_losses
Нtrainable_variables

0
	1
 

0
	1
Е
цmetrics
П	variables
чlayers
шnon_trainable_variables
щlayer_metrics
Рregularization_losses
 ъlayer_regularization_losses
Сtrainable_variables


0
1
 


0
1
Е
ыmetrics
У	variables
ьlayers
эnon_trainable_variables
юlayer_metrics
Фregularization_losses
 яlayer_regularization_losses
Хtrainable_variables


kernel
recurrent_kernel
bias
№	variables
ёregularization_losses
ђtrainable_variables
ѓ	keras_api


kernel
recurrent_kernel
bias
є	variables
ѕregularization_losses
іtrainable_variables
ї	keras_api
 

Ч0
Ш1
 
 
 
 
 
 
Е
јmetrics
Ю	variables
љlayers
њnon_trainable_variables
ћlayer_metrics
Яregularization_losses
 ќlayer_regularization_losses
аtrainable_variables

0
1
 

0
1
Е
§metrics
в	variables
ўlayers
џnon_trainable_variables
layer_metrics
гregularization_losses
 layer_regularization_losses
дtrainable_variables

0
1
 

0
1
Е
metrics
ж	variables
layers
non_trainable_variables
layer_metrics
зregularization_losses
 layer_regularization_losses
иtrainable_variables


kernel
recurrent_kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api


kernel
recurrent_kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
 

к0
л1
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
 
 
 

0
1
2
 

0
1
2
Е
metrics
№	variables
layers
non_trainable_variables
layer_metrics
ёregularization_losses
 layer_regularization_losses
ђtrainable_variables

0
1
2
 

0
1
2
Е
metrics
є	variables
layers
non_trainable_variables
layer_metrics
ѕregularization_losses
 layer_regularization_losses
іtrainable_variables
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

0
1
2
 

0
1
2
Е
metrics
	variables
layers
non_trainable_variables
layer_metrics
regularization_losses
 layer_regularization_losses
trainable_variables

0
1
2
 

0
1
2
Е
metrics
	variables
layers
 non_trainable_variables
Ёlayer_metrics
regularization_losses
 Ђlayer_regularization_losses
trainable_variables
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
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
w
action_0/observationPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
j
action_0/rewardPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m
action_0/step_typePlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

 action_1/actor_network_state/0/0Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

 action_1/actor_network_state/0/1Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

 action_1/actor_network_state/1/0Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

 action_1/actor_network_state/1/1Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
и
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type action_1/actor_network_state/0/0 action_1/actor_network_state/0/1 action_1/actor_network_state/1/0 action_1/actor_network_state/1/1TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/biasWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernelaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernelUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/biasYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernelcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernelWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/biasQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/biasCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias* 
Tin
2*
Tout	
2*
_collective_manager_ids
 *w
_output_shapese
c:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU2	 *0,1J 8 *-
f(R&
$__inference_signature_wrapper_146237
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ь
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *-
f(R&
$__inference_signature_wrapper_146250
с
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *-
f(R&
$__inference_signature_wrapper_146262

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
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *-
f(R&
$__inference_signature_wrapper_146258
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameglobal_step/Read/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernel/Read/ReadVariableOpuActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernel/Read/ReadVariableOpiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/bias/Read/ReadVariableOpmActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernel/Read/ReadVariableOpwActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpkActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/bias/Read/ReadVariableOpWActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpeActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOpcActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpRValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpPValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpRValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpPValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpWValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/kernel/Read/ReadVariableOpaValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpUValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/bias/Read/ReadVariableOpWValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/kernel/Read/ReadVariableOpaValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpUValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/bias/Read/ReadVariableOp2ValueRnnNetwork/dense_4/kernel/Read/ReadVariableOp0ValueRnnNetwork/dense_4/bias/Read/ReadVariableOpConst*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *(
f#R!
__inference__traced_save_146380

StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameglobal_stepTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernelTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/biasWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernelaActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernelUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/biasYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernelcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernelWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/biasCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/biasQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/biasCValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/kernelMValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/recurrent_kernelAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/biasCValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/kernelMValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/recurrent_kernelAValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/biasValueRnnNetwork/dense_4/kernelValueRnnNetwork/dense_4/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *+
f&R$
"__inference__traced_restore_146468У
н

$__inference_signature_wrapper_146237
discount
observation

reward
	step_type
actor_network_state_0_0
actor_network_state_0_1
actor_network_state_1_0
actor_network_state_1_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity

identity_1

identity_2

identity_3

identity_4ЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationactor_network_state_0_0actor_network_state_0_1actor_network_state_1_0actor_network_state_1_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11* 
Tin
2*
Tout	
2*
_collective_manager_ids
 *w
_output_shapese
c:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU2	 *0,1J 8 *3
f.R,
*__inference_function_with_signature_1167192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*й
_input_shapesЧ
Ф:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_name0/observation:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:c_
(
_output_shapes
:џџџџџџџџџ
3
_user_specified_name1/actor_network_state/0/0:c_
(
_output_shapes
:џџџџџџџџџ
3
_user_specified_name1/actor_network_state/0/1:c_
(
_output_shapes
:џџџџџџџџџ
3
_user_specified_name1/actor_network_state/1/0:c_
(
_output_shapes
:џџџџџџџџџ
3
_user_specified_name1/actor_network_state/1/1
§
d
*__inference_function_with_signature_116827
unknown
identity	ЂStatefulPartitionedCallЉ
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
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *"
fR
__inference_<lambda>_11892
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall

^
$__inference_signature_wrapper_146258
unknown
identity	ЂStatefulPartitionedCallК
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
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *3
f.R,
*__inference_function_with_signature_1168272
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
Ь
&
$__inference_signature_wrapper_146262
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
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *3
f.R,
*__inference_function_with_signature_1168382
PartitionedCall*
_input_shapes 
в

*__inference_function_with_signature_116719
	step_type

reward
discount
observation
actor_network_state_0_0
actor_network_state_0_1
actor_network_state_1_0
actor_network_state_1_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity

identity_1

identity_2

identity_3

identity_4ЂStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationactor_network_state_0_0actor_network_state_0_1actor_network_state_1_0actor_network_state_1_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11* 
Tin
2*
Tout	
2*
_collective_manager_ids
 *w
_output_shapese
c:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*/
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU2	 *0,1J 8 *"
fR
__inference_action_1166822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*й
_input_shapesЧ
Ф:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_name0/observation:c_
(
_output_shapes
:џџџџџџџџџ
3
_user_specified_name1/actor_network_state/0/0:c_
(
_output_shapes
:џџџџџџџџџ
3
_user_specified_name1/actor_network_state/0/1:c_
(
_output_shapes
:џџџџџџџџџ
3
_user_specified_name1/actor_network_state/1/0:c_
(
_output_shapes
:џџџџџџџџџ
3
_user_specified_name1/actor_network_state/1/1
С
,
*__inference_function_with_signature_116838§
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
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *"
fR
__inference_<lambda>_11922
PartitionedCall*
_input_shapes 
їП
њ
__inference_action_116682
	time_step
time_step_1
time_step_2
time_step_3
policy_state
policy_state_1
policy_state_2
policy_state_3p
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resourceq
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resourcer
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resources
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourcem
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resourcen
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resourceb
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4ЂdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЂcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂyActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpЂxActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpЂzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpЂ{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpЂzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpЂ|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpЂUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpЂaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpЂ`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpI
ShapeShapetime_step_2*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedm
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constp
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosq
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constx
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_1q
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2packed:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constx
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_2q
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constx
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_3T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	time_stepEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis
concat_4ConcatV2Shape_1:output:0ones:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4m
ReshapeReshape	Equal:z:0concat_4:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2	
Reshape
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_1

SelectV2_2SelectV2Reshape:output:0zeros_2:output:0policy_state_2*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_2

SelectV2_3SelectV2Reshape:output:0zeros_3:output:0policy_state_3*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_3M
Shape_2Shapetime_step_2*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1q
shape_as_tensor_4Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_4`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis
concat_5ConcatV2packed_1:output:0shape_as_tensor_4:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5c
zeros_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_4/Constx
zeros_4Fillconcat_5:output:0zeros_4/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_4q
shape_as_tensor_5Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_5`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_6/axis
concat_6ConcatV2packed_1:output:0shape_as_tensor_5:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:2

concat_6c
zeros_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_5/Constx
zeros_5Fillconcat_6:output:0zeros_5/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_5q
shape_as_tensor_6Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_6`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_7/axis
concat_7ConcatV2packed_1:output:0shape_as_tensor_6:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:2

concat_7c
zeros_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_6/Constx
zeros_6Fillconcat_7:output:0zeros_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_6q
shape_as_tensor_7Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_7`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_8/axis
concat_8ConcatV2packed_1:output:0shape_as_tensor_7:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:2

concat_8c
zeros_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_7/Constx
zeros_7Fillconcat_8:output:0zeros_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_7X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	time_stepEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_9/axis
concat_9ConcatV2Shape_3:output:0ones_1:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes
:2

concat_9s
	Reshape_1ReshapeEqual_1:z:0concat_9:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1

SelectV2_4SelectV2Reshape_1:output:0zeros_4:output:0SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_4

SelectV2_5SelectV2Reshape_1:output:0zeros_5:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_5

SelectV2_6SelectV2Reshape_1:output:0zeros_6:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_6

SelectV2_7SelectV2Reshape_1:output:0zeros_7:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_7в
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimЊ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_3OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2D
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDimsж
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimЊ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2F
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1Х
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2]
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeЖ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshapeџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstЗ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshapeј
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpИ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulї
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02f
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpК
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddУ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Reluў
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpН
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulќ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpС
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddШ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluЄ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackЈ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Ј
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceр
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeЈ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackЌ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Ќ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2ј
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisЪ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2`
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatо
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџd2a
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeТ
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yЯ
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskм
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2M
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rankъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaб
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0ъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisэ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatЉ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџd2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose 
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1ы
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulэ
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/LessLessTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul:z:0\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Lessё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedэ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosя
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3Ъ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџd*
squeeze_dims
 2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeЬ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1Ђ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectЈ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_5:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1Ј
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2:output:0SelectV2_6:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2Ј
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3:output:0SelectV2_7:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3И
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype02z
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpя
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMulП
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02|
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpє
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1ј
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/addAddV2sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul:product:0uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/addЗ
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02{
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAddBiasAddjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/ConstЊ
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2t
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dimЯ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/splitSplit{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dim:output:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/SigmoidSigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1SigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1ж
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mulMulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/TanhTanhqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanhы
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1MulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid:y:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1ъ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1AddV2jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul:z:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2SigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2џ
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1TanhlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1я
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2MulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2П
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02|
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMulMatMullActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMulХ
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02~
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpќ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/addAddV2uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul:product:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/addН
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02}
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAddlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/ConstЎ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2v
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dimз
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/splitSplit}ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1SigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2p
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1м
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mulMulrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/TanhTanhsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanhѓ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1MulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1ђ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1AddV2lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul:z:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2SigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2p
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1TanhnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1ї
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2MulrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2№
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimЛ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDimsГ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2A
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeezeя
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02b
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2S
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulэ
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02c
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp­
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2T
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddз
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2C
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeр
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshapeє
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2:
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhЛ
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x­
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulЛ
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xЎ
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addќ
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2@
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_likeЩ
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp№
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2H
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddл
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2E
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeк
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstЎ
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2r
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeс
АActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2Г
АActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shapeж
ОActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2С
ОActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackб
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2У
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1б
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2У
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ы	
ИActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceЙActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ЧActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ЩActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ЩActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2Л
ИActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceѕ
КActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackСActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2Н
КActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Е
ЖActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Й
ЖActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisч
БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2ЙActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0УActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ПActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2Д
БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Ё
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ2Ѓ
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ѓ
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Ъ
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceКActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ЉActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ЉActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceУ
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeО
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЫ
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Т
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2С
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicesActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceЖ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgsЁActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0{ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsѕ
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstД
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFillwActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0_ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2R
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosч
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesЏ
ЏActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2В
ЏActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeф
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroЌ
ЮActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeYActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2б
ЮActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shapeќ
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2м
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1
жActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsзActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0тActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1:output:0*
_output_shapes
:2й
жActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsО
дActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityлActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2з
дActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeі	
яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityнActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2ђ
яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЅ
ЙActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentityјActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2М
ЙActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeЛ
ЕActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2И
ЕActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Constя
ЛActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentityОActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2О
ЛActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeќ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisм
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2ТActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0ФActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatЏ
XActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/IdentityIdentity;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2Z
XActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identityџ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstЏ
UActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosFill_ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2W
UActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosЈ
SActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addAddV2aActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity:output:0^ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2U
SActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addm
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeУ
Deterministic_1/sample/ShapeShapeWActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1в
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЈ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat
"Deterministic_1/sample/BroadcastToBroadcastToWActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0&Deterministic_1/sample/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1д
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/yЖ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
clip_by_value
IdentityIdentityclip_by_value:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity№

Identity_1IdentitylActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1№

Identity_2IdentitylActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2ђ

Identity_3IdentitynActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3ђ

Identity_4IdentitynActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*й
_input_shapesЧ
Ф:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::2Ь
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2Ъ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2і
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpyActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2є
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpxActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2ј
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2њ
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2ј
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2ќ
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2Ў
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2Ц
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2Ф
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:VR
(
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state:VR
(
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state:VR
(
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state:VR
(
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state
	
t
$__inference_signature_wrapper_146250

batch_size
identity

identity_1

identity_2

identity_3ѓ
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *3
f.R,
*__inference_function_with_signature_1168072
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1q

Identity_2IdentityPartitionedCall:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2q

Identity_3IdentityPartitionedCall:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
п
t
$__inference_get_initial_state_116798

batch_size
identity

identity_1

identity_2

identity_3R
packedPack
batch_size*
N*
T0*
_output_shapes
:2
packedm
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constp
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosq
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constx
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_1q
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2packed:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constx
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_2q
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constx
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_3c
IdentityIdentityzeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1Identityzeros_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1i

Identity_2Identityzeros_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2i

Identity_3Identityzeros_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
	
z
*__inference_function_with_signature_116807

batch_size
identity

identity_1

identity_2

identity_3э
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU2	 *0,1J 8 *-
f(R&
$__inference_get_initial_state_1167982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1q

Identity_2IdentityPartitionedCall:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2q

Identity_3IdentityPartitionedCall:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
0

__inference_<lambda>_1192*
_input_shapes 
ЄІ
Ё
"__inference_distribution_fn_117804
	step_type

reward
discount
observation
actor_network_state_0_0
actor_network_state_0_1
actor_network_state_1_0
actor_network_state_1_1p
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resourceq
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resourcer
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resources
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourcem
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resourcen
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resourceb
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4ЂdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЂcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂyActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpЂxActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpЂzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpЂ{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpЂzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpЂ|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpЂUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpЂaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpЂ`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpF
ShapeShapediscount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedm
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constp
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosq
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constx
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_1q
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2packed:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constx
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_2q
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constx
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_3T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis
concat_4ConcatV2Shape_1:output:0ones:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4m
ReshapeReshape	Equal:z:0concat_4:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2	
Reshape
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0_0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_0_1*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_1

SelectV2_2SelectV2Reshape:output:0zeros_2:output:0actor_network_state_1_0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_2

SelectV2_3SelectV2Reshape:output:0zeros_3:output:0actor_network_state_1_1*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_3J
Shape_2Shapediscount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1q
shape_as_tensor_4Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_4`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis
concat_5ConcatV2packed_1:output:0shape_as_tensor_4:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5c
zeros_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_4/Constx
zeros_4Fillconcat_5:output:0zeros_4/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_4q
shape_as_tensor_5Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_5`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_6/axis
concat_6ConcatV2packed_1:output:0shape_as_tensor_5:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:2

concat_6c
zeros_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_5/Constx
zeros_5Fillconcat_6:output:0zeros_5/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_5q
shape_as_tensor_6Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_6`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_7/axis
concat_7ConcatV2packed_1:output:0shape_as_tensor_6:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:2

concat_7c
zeros_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_6/Constx
zeros_6Fillconcat_7:output:0zeros_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_6q
shape_as_tensor_7Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_7`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_8/axis
concat_8ConcatV2packed_1:output:0shape_as_tensor_7:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:2

concat_8c
zeros_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_7/Constx
zeros_7Fillconcat_8:output:0zeros_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_7X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_9/axis
concat_9ConcatV2Shape_3:output:0ones_1:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes
:2

concat_9s
	Reshape_1ReshapeEqual_1:z:0concat_9:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1

SelectV2_4SelectV2Reshape_1:output:0zeros_4:output:0SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_4

SelectV2_5SelectV2Reshape_1:output:0zeros_5:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_5

SelectV2_6SelectV2Reshape_1:output:0zeros_6:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_6

SelectV2_7SelectV2Reshape_1:output:0zeros_7:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_7в
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimЊ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2D
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDimsж
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimЊ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2F
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1Х
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2]
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeЖ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshapeџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstЗ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshapeј
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpИ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulї
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02f
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpК
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddУ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Reluў
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpН
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulќ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpС
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddШ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluЄ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackЈ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Ј
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceр
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeЈ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackЌ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Ќ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2ј
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisЪ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2`
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatо
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџd2a
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeТ
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yЯ
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskм
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2M
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rankъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaб
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0ъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisэ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatЉ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџd2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose 
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1ы
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulэ
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/LessLessTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul:z:0\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Lessё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedэ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosя
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3Ъ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџd*
squeeze_dims
 2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeЬ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1Ђ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectЈ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_5:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1Ј
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2:output:0SelectV2_6:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2Ј
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3:output:0SelectV2_7:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3И
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype02z
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpя
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMulП
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02|
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpє
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1ј
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/addAddV2sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul:product:0uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/addЗ
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02{
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAddBiasAddjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/ConstЊ
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2t
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dimЯ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/splitSplit{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dim:output:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/SigmoidSigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1SigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1ж
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mulMulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/TanhTanhqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanhы
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1MulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid:y:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1ъ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1AddV2jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul:z:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2SigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2џ
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1TanhlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1я
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2MulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2П
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02|
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMulMatMullActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMulХ
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02~
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpќ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/addAddV2uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul:product:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/addН
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02}
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAddlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/ConstЎ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2v
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dimз
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/splitSplit}ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1SigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2p
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1м
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mulMulrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/TanhTanhsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanhѓ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1MulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1ђ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1AddV2lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul:z:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2SigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2p
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1TanhnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1ї
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2MulrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2№
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimЛ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDimsГ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2A
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeezeя
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02b
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2S
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulэ
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02c
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp­
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2T
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddз
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2C
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeр
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshapeє
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2:
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhЛ
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x­
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulЛ
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xЎ
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addќ
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2@
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_likeЩ
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp№
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2H
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddл
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2E
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeк
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstЎ
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2r
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeс
АActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2Г
АActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shapeж
ОActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2С
ОActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackб
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2У
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1б
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2У
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ы	
ИActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceЙActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ЧActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ЩActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ЩActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2Л
ИActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceѕ
КActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackСActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2Н
КActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Е
ЖActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Й
ЖActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisч
БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2ЙActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0УActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ПActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2Д
БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Ё
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ2Ѓ
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ѓ
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Ъ
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceКActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ЉActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ЉActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceУ
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeО
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЫ
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Т
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2С
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicesActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceЖ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgsЁActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0{ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsѕ
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstД
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFillwActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0_ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2R
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosч
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesЏ
ЏActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2В
ЏActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeф
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroЌ
ЮActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeYActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2б
ЮActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shapeќ
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2м
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1
жActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsзActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0тActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1:output:0*
_output_shapes
:2й
жActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsО
дActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityлActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2з
дActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeі	
яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityнActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2ђ
яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЅ
ЙActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentityјActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2М
ЙActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeЛ
ЕActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2И
ЕActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Constя
ЛActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentityОActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2О
ЛActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeќ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisм
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2ТActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0ФActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatЏ
XActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/IdentityIdentity;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2Z
XActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identityџ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstЏ
UActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosFill_ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2W
UActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosЈ
SActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addAddV2aActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity:output:0^ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2U
SActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addm
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtolq
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic_1/atolq
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic_1/rtolж
IdentityIdentityWActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity№

Identity_1IdentitylActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1№

Identity_2IdentitylActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2ђ

Identity_3IdentitynActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3ђ

Identity_4IdentitynActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_4q
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic_2/atolq
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic_2/rtol"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*й
_input_shapesЧ
Ф:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::2Ь
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2Ъ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2і
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpyActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2є
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpxActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2ј
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2њ
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2ј
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2ќ
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2Ў
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2Ц
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2Ф
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameobservation:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameactor_network_state/0/0:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameactor_network_state/0/1:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameactor_network_state/1/0:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameactor_network_state/1/1
пР

__inference_action_117495
	step_type

reward
discount
observation
actor_network_state_0_0
actor_network_state_0_1
actor_network_state_1_0
actor_network_state_1_1p
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resourceq
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resourcer
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resources
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourcem
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resourcen
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resourceb
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4ЂdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЂcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂyActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpЂxActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpЂzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpЂ{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpЂzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpЂ|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpЂUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpЂaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpЂ`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpF
ShapeShapediscount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedm
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constp
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosq
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constx
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_1q
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2packed:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constx
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_2q
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constx
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_3T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis
concat_4ConcatV2Shape_1:output:0ones:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4m
ReshapeReshape	Equal:z:0concat_4:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2	
Reshape
SelectV2SelectV2Reshape:output:0zeros:output:0actor_network_state_0_0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0actor_network_state_0_1*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_1

SelectV2_2SelectV2Reshape:output:0zeros_2:output:0actor_network_state_1_0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_2

SelectV2_3SelectV2Reshape:output:0zeros_3:output:0actor_network_state_1_1*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_3J
Shape_2Shapediscount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1q
shape_as_tensor_4Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_4`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis
concat_5ConcatV2packed_1:output:0shape_as_tensor_4:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5c
zeros_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_4/Constx
zeros_4Fillconcat_5:output:0zeros_4/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_4q
shape_as_tensor_5Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_5`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_6/axis
concat_6ConcatV2packed_1:output:0shape_as_tensor_5:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:2

concat_6c
zeros_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_5/Constx
zeros_5Fillconcat_6:output:0zeros_5/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_5q
shape_as_tensor_6Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_6`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_7/axis
concat_7ConcatV2packed_1:output:0shape_as_tensor_6:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:2

concat_7c
zeros_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_6/Constx
zeros_6Fillconcat_7:output:0zeros_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_6q
shape_as_tensor_7Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_7`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_8/axis
concat_8ConcatV2packed_1:output:0shape_as_tensor_7:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:2

concat_8c
zeros_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_7/Constx
zeros_7Fillconcat_8:output:0zeros_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_7X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_9/axis
concat_9ConcatV2Shape_3:output:0ones_1:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes
:2

concat_9s
	Reshape_1ReshapeEqual_1:z:0concat_9:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1

SelectV2_4SelectV2Reshape_1:output:0zeros_4:output:0SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_4

SelectV2_5SelectV2Reshape_1:output:0zeros_5:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_5

SelectV2_6SelectV2Reshape_1:output:0zeros_6:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_6

SelectV2_7SelectV2Reshape_1:output:0zeros_7:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_7в
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimЊ
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2D
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDimsж
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimЊ
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2F
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1Х
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2]
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeЖ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshapeџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstЗ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshapeј
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpИ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulї
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02f
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpК
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddУ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Reluў
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpН
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulќ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpС
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddШ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluЄ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackЈ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Ј
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceр
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeЈ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackЌ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Ќ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2ј
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisЪ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2`
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatо
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџd2a
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeТ
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yЯ
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskм
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2M
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rankъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaб
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0ъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisэ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatЉ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџd2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose 
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1ы
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulэ
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/LessLessTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul:z:0\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Lessё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedэ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosя
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3Ъ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџd*
squeeze_dims
 2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeЬ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1Ђ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectЈ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_5:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1Ј
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2:output:0SelectV2_6:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2Ј
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3:output:0SelectV2_7:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3И
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype02z
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpя
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMulП
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02|
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpє
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1ј
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/addAddV2sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul:product:0uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/addЗ
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02{
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAddBiasAddjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/ConstЊ
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2t
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dimЯ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/splitSplit{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dim:output:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/SigmoidSigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1SigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1ж
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mulMulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/TanhTanhqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanhы
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1MulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid:y:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1ъ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1AddV2jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul:z:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2SigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2џ
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1TanhlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1я
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2MulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2П
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02|
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMulMatMullActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMulХ
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02~
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpќ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/addAddV2uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul:product:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/addН
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02}
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAddlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/ConstЎ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2v
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dimз
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/splitSplit}ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1SigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2p
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1м
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mulMulrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/TanhTanhsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanhѓ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1MulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1ђ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1AddV2lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul:z:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2SigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2p
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1TanhnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1ї
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2MulrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2№
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimЛ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDimsГ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2A
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeezeя
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02b
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2S
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulэ
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02c
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp­
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2T
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddз
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2C
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeр
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshapeє
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2:
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhЛ
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x­
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulЛ
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xЎ
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addќ
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2@
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_likeЩ
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp№
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2H
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddл
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2E
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeк
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstЎ
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2r
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeс
АActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2Г
АActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shapeж
ОActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2С
ОActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackб
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2У
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1б
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2У
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ы	
ИActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceЙActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ЧActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ЩActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ЩActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2Л
ИActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceѕ
КActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackСActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2Н
КActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Е
ЖActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Й
ЖActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisч
БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2ЙActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0УActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ПActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2Д
БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Ё
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ2Ѓ
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ѓ
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Ъ
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceКActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ЉActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ЉActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceУ
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeО
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЫ
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Т
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2С
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicesActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceЖ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgsЁActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0{ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsѕ
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstД
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFillwActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0_ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2R
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosч
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesЏ
ЏActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2В
ЏActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeф
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroЌ
ЮActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeYActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2б
ЮActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shapeќ
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2м
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1
жActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsзActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0тActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1:output:0*
_output_shapes
:2й
жActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsО
дActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityлActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2з
дActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeі	
яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityнActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2ђ
яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЅ
ЙActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentityјActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2М
ЙActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeЛ
ЕActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2И
ЕActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Constя
ЛActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentityОActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2О
ЛActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeќ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisм
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2ТActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0ФActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatЏ
XActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/IdentityIdentity;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2Z
XActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identityџ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstЏ
UActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosFill_ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2W
UActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosЈ
SActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addAddV2aActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity:output:0^ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2U
SActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addm
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeУ
Deterministic_1/sample/ShapeShapeWActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1в
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЈ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat
"Deterministic_1/sample/BroadcastToBroadcastToWActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0&Deterministic_1/sample/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1д
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/yЖ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
clip_by_value
IdentityIdentityclip_by_value:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity№

Identity_1IdentitylActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1№

Identity_2IdentitylActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2ђ

Identity_3IdentitynActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3ђ

Identity_4IdentitynActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*й
_input_shapesЧ
Ф:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::2Ь
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2Ъ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2і
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpyActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2є
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpxActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2ј
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2њ
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2ј
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2ќ
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2Ў
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2Ц
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2Ф
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameobservation:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameactor_network_state/0/0:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameactor_network_state/0/1:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameactor_network_state/1/0:a]
(
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameactor_network_state/1/1
А
П
"__inference__traced_restore_146468
file_prefix 
assignvariableop_global_stepk
gassignvariableop_1_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kerneli
eassignvariableop_2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasm
iassignvariableop_3_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernelk
gassignvariableop_4_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasn
jassignvariableop_5_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_kernelx
tassignvariableop_6_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_recurrent_kernell
hassignvariableop_7_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasp
lassignvariableop_8_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_kernelz
vassignvariableop_9_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_recurrent_kernelo
kassignvariableop_10_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_bias[
Wassignvariableop_11_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasi
eassignvariableop_12_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernelg
cassignvariableop_13_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasV
Rassignvariableop_14_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_kernelT
Passignvariableop_15_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_biasV
Rassignvariableop_16_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernelT
Passignvariableop_17_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias[
Wassignvariableop_18_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_kernele
aassignvariableop_19_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_recurrent_kernelY
Uassignvariableop_20_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_bias[
Wassignvariableop_21_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_kernele
aassignvariableop_22_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_recurrent_kernelY
Uassignvariableop_23_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_bias6
2assignvariableop_24_valuernnnetwork_dense_4_kernel4
0assignvariableop_25_valuernnnetwork_dense_4_bias
identity_27ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9л

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ч	
valueн	Bк	B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesФ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesГ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_global_stepIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ь
AssignVariableOp_1AssignVariableOpgassignvariableop_1_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ъ
AssignVariableOp_2AssignVariableOpeassignvariableop_2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ю
AssignVariableOp_3AssignVariableOpiassignvariableop_3_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ь
AssignVariableOp_4AssignVariableOpgassignvariableop_4_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5я
AssignVariableOp_5AssignVariableOpjassignvariableop_5_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6љ
AssignVariableOp_6AssignVariableOptassignvariableop_6_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7э
AssignVariableOp_7AssignVariableOphassignvariableop_7_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ё
AssignVariableOp_8AssignVariableOplassignvariableop_8_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ћ
AssignVariableOp_9AssignVariableOpvassignvariableop_9_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ѓ
AssignVariableOp_10AssignVariableOpkassignvariableop_10_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11п
AssignVariableOp_11AssignVariableOpWassignvariableop_11_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12э
AssignVariableOp_12AssignVariableOpeassignvariableop_12_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ы
AssignVariableOp_13AssignVariableOpcassignvariableop_13_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14к
AssignVariableOp_14AssignVariableOpRassignvariableop_14_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15и
AssignVariableOp_15AssignVariableOpPassignvariableop_15_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16к
AssignVariableOp_16AssignVariableOpRassignvariableop_16_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17и
AssignVariableOp_17AssignVariableOpPassignvariableop_17_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18п
AssignVariableOp_18AssignVariableOpWassignvariableop_18_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19щ
AssignVariableOp_19AssignVariableOpaassignvariableop_19_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_recurrent_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20н
AssignVariableOp_20AssignVariableOpUassignvariableop_20_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21п
AssignVariableOp_21AssignVariableOpWassignvariableop_21_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22щ
AssignVariableOp_22AssignVariableOpaassignvariableop_22_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_recurrent_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23н
AssignVariableOp_23AssignVariableOpUassignvariableop_23_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24К
AssignVariableOp_24AssignVariableOp2assignvariableop_24_valuernnnetwork_dense_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25И
AssignVariableOp_25AssignVariableOp0assignvariableop_25_valuernnnetwork_dense_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_27"#
identity_27Identity_27:output:0*}
_input_shapesl
j: ::::::::::::::::::::::::::2$
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
AssignVariableOp_25AssignVariableOp_252(
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
п
t
$__inference_get_initial_state_117832

batch_size
identity

identity_1

identity_2

identity_3R
packedPack
batch_size*
N*
T0*
_output_shapes
:2
packedm
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constp
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosq
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constx
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_1q
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2packed:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constx
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_2q
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constx
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_3c
IdentityIdentityzeros:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1Identityzeros_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1i

Identity_2Identityzeros_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2i

Identity_3Identityzeros_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
У
є
__inference_action_117169
time_step_step_type
time_step_reward
time_step_discount
time_step_observation(
$policy_state_actor_network_state_0_0(
$policy_state_actor_network_state_0_1(
$policy_state_actor_network_state_1_0(
$policy_state_actor_network_state_1_1p
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resourceq
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resourcer
nactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resources
oactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource
actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourcem
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resourcen
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resourceb
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4ЂdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЂcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЂfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂyActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpЂxActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpЂzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpЂ{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpЂzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpЂ|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpЂUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpЂaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpЂ`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpP
ShapeShapetime_step_discount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedm
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constp
zerosFillconcat:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosq
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constx
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_1q
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2packed:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constx
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_2q
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constx
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_3T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yl
EqualEqualtime_step_step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis
concat_4ConcatV2Shape_1:output:0ones:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4m
ReshapeReshape	Equal:z:0concat_4:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2	
Reshape
SelectV2SelectV2Reshape:output:0zeros:output:0$policy_state_actor_network_state_0_0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2Ё

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0$policy_state_actor_network_state_0_1*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_1Ё

SelectV2_2SelectV2Reshape:output:0zeros_2:output:0$policy_state_actor_network_state_1_0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_2Ё

SelectV2_3SelectV2Reshape:output:0zeros_3:output:0$policy_state_actor_network_state_1_1*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_3T
Shape_2Shapetime_step_discount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1q
shape_as_tensor_4Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_4`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis
concat_5ConcatV2packed_1:output:0shape_as_tensor_4:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5c
zeros_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_4/Constx
zeros_4Fillconcat_5:output:0zeros_4/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_4q
shape_as_tensor_5Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_5`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_6/axis
concat_6ConcatV2packed_1:output:0shape_as_tensor_5:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:2

concat_6c
zeros_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_5/Constx
zeros_5Fillconcat_6:output:0zeros_5/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_5q
shape_as_tensor_6Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_6`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_7/axis
concat_7ConcatV2packed_1:output:0shape_as_tensor_6:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:2

concat_7c
zeros_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_6/Constx
zeros_6Fillconcat_7:output:0zeros_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_6q
shape_as_tensor_7Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_7`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_8/axis
concat_8ConcatV2packed_1:output:0shape_as_tensor_7:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:2

concat_8c
zeros_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_7/Constx
zeros_7Fillconcat_8:output:0zeros_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
zeros_7X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yr
Equal_1Equaltime_step_step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_9/axis
concat_9ConcatV2Shape_3:output:0ones_1:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes
:2

concat_9s
	Reshape_1ReshapeEqual_1:z:0concat_9:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1

SelectV2_4SelectV2Reshape_1:output:0zeros_4:output:0SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_4

SelectV2_5SelectV2Reshape_1:output:0zeros_5:output:0SelectV2_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_5

SelectV2_6SelectV2Reshape_1:output:0zeros_6:output:0SelectV2_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_6

SelectV2_7SelectV2Reshape_1:output:0zeros_7:output:0SelectV2_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

SelectV2_7в
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimД
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_observationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2D
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDimsж
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimД
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2F
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1Х
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	2]
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeЖ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshapeџ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstЗ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshapeј
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype02e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpИ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulї
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02f
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpК
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddУ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Reluў
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpnactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpН
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMulќ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpС
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAddШ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluRelu`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/ReluЄ
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackЈ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Ј
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2g
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceр
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	2_
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeЈ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackЌ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Ќ
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2q
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2ј
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2e
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisЪ
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:2`
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatо
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshapebActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџd2a
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeТ
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yЯ
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskм
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2M
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rankъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaб
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2X
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0ъ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisэ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatЉ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџd2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose 
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2\
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1ы
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mulэ
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/LessLessTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/mul:z:0\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Lessё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedэ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2N
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosя
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2я
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/y
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mulMul]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2T
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mulё
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/y
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/LessLessVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/mul:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2U
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Lessѕ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2Y
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1Ѕ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packedё
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2V
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Const
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3Ъ
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџd*
squeeze_dims
 2P
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeЬ
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 2R
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1Ђ
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2O
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectЈ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_5:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1Ј
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_2:output:0SelectV2_6:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2Ј
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_3:output:0SelectV2_7:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2Q
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3И
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype02z
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpя
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMulП
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02|
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpє
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1ј
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/addAddV2sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul:product:0uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/addЗ
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02{
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAddBiasAddjActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/ConstЊ
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2t
rActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dimЯ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/splitSplit{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split/split_dim:output:0sActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/SigmoidSigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1SigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1ж
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mulMulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/TanhTanhqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2i
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanhы
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1MulnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid:y:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1ъ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1AddV2jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul:z:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2SigmoidqActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2џ
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1TanhlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1я
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2MulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2П
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02|
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMulMatMullActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMulХ
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02~
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpќ
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_2:output:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2o
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/addAddV2uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul:product:0wActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/addН
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02}
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAddlActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add:z:0ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/ConstЎ
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2v
tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dimз
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/splitSplit}ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0uActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2n
lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1SigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2p
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1м
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mulMulrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/TanhTanhsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:џџџџџџџџџ2k
iActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanhѓ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1MulpActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1ђ
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1AddV2lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul:z:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2SigmoidsActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:џџџџџџџџџ2p
nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1TanhnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2m
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1ї
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2MulrActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2l
jActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2№
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2W
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimЛ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsnActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2S
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDimsГ
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2A
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeezeя
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02b
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2S
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulэ
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02c
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp­
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2T
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddз
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2C
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeр
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshapeє
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2:
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhЛ
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x­
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulЛ
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xЎ
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addќ
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2@
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_likeЩ
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp№
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2H
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddл
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2E
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeк
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstЎ
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2r
pActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeс
АActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2Г
АActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shapeж
ОActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2С
ОActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackб
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2У
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1б
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2У
РActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ы	
ИActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceЙActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ЧActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ЩActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ЩActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2Л
ИActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceѕ
КActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackСActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2Н
КActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Е
ЖActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Й
ЖActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisч
БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2ЙActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0УActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ПActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2Д
БActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Ё
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ2Ѓ
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ѓ
 ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Ъ
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceКActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ЇActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ЉActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ЉActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceУ
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2l
jActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeО
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2z
xActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЫ
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Т
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2|
zActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2С
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicesActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceЖ
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgsЁActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0{ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2t
rActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsѕ
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
VActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstД
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFillwActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0_ActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2R
PActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosч
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesЏ
ЏActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2В
ЏActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeф
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Q
OActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroЌ
ЮActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeYActorDistributionRnnNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2б
ЮActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shapeќ
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2м
йActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1
жActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsзActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0тActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1:output:0*
_output_shapes
:2й
жActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsО
дActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityлActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2з
дActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeі	
яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentityнActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2ђ
яActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeЅ
ЙActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentityјActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2М
ЙActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeЛ
ЕActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2И
ЕActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Constя
ЛActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentityОActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2О
ЛActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeќ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisм
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2ТActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0ФActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2X
VActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatЏ
XActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/IdentityIdentity;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2Z
XActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identityџ
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2]
[ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstЏ
UActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosFill_ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0dActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2W
UActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosЈ
SActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addAddV2aActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity:output:0^ActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2U
SActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addm
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeУ
Deterministic_1/sample/ShapeShapeWActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1в
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЈ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat
"Deterministic_1/sample/BroadcastToBroadcastToWActorDistributionRnnNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0&Deterministic_1/sample/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1д
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/yЖ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
clip_by_value
IdentityIdentityclip_by_value:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity№

Identity_1IdentitylActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/mul_2:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1№

Identity_2IdentitylActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/add_1:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_2ђ

Identity_3IdentitynActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/mul_2:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_3ђ

Identity_4IdentitynActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/add_1:z:0e^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpf^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpz^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpy^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp|^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp}^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*й
_input_shapesЧ
Ф:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::2Ь
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2Ъ
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2а
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2Ю
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpeActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2і
yActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpyActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2є
xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpxActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2ј
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2њ
{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp{ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2ј
zActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpzActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2ќ
|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp|ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2Ў
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2Ц
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2Ф
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:X T
#
_output_shapes
:џџџџџџџџџ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:џџџџџџџџџ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:џџџџџџџџџ
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nametime_step/observation:nj
(
_output_shapes
:џџџџџџџџџ
>
_user_specified_name&$policy_state/actor_network_state/0/0:nj
(
_output_shapes
:џџџџџџџџџ
>
_user_specified_name&$policy_state/actor_network_state/0/1:nj
(
_output_shapes
:џџџџџџџџџ
>
_user_specified_name&$policy_state/actor_network_state/1/0:nj
(
_output_shapes
:џџџџџџџџџ
>
_user_specified_name&$policy_state/actor_network_state/1/1

Z
__inference_<lambda>_1189
readvariableop_resource
identity	ЂReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2 
ReadVariableOpReadVariableOp
ДO
і
__inference__traced_save_146380
file_prefix*
&savev2_global_step_read_readvariableop	s
osavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel_read_readvariableopq
msavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias_read_readvariableopu
qsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableops
osavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableopv
rsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_kernel_read_readvariableop
|savev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_recurrent_kernel_read_readvariableopt
psavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_bias_read_readvariableopx
tsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_kernel_read_readvariableop
~savev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_recurrent_kernel_read_readvariableopv
rsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_bias_read_readvariableopb
^savev2_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias_read_readvariableopp
lsavev2_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopn
jsavev2_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableop]
Ysavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_kernel_read_readvariableop[
Wsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_bias_read_readvariableop]
Ysavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel_read_readvariableop[
Wsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias_read_readvariableopb
^savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_kernel_read_readvariableopl
hsavev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_recurrent_kernel_read_readvariableop`
\savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_bias_read_readvariableopb
^savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_kernel_read_readvariableopl
hsavev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_recurrent_kernel_read_readvariableop`
\savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_bias_read_readvariableop=
9savev2_valuernnnetwork_dense_4_kernel_read_readvariableop;
7savev2_valuernnnetwork_dense_4_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameе

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ч	
valueн	Bк	B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesО
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesє
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_global_step_read_readvariableoposavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel_read_readvariableopmsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias_read_readvariableopqsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableoposavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableoprsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_kernel_read_readvariableop|savev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_recurrent_kernel_read_readvariableoppsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_bias_read_readvariableoptsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_kernel_read_readvariableop~savev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_recurrent_kernel_read_readvariableoprsavev2_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_1_bias_read_readvariableop^savev2_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias_read_readvariableoplsavev2_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopjsavev2_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableopYsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_kernel_read_readvariableopWsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_2_bias_read_readvariableopYsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_kernel_read_readvariableopWsavev2_valuernnnetwork_valuernnnetwork_encodingnetwork_dense_3_bias_read_readvariableop^savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_kernel_read_readvariableophsavev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_recurrent_kernel_read_readvariableop\savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_2_bias_read_readvariableop^savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_kernel_read_readvariableophsavev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_recurrent_kernel_read_readvariableop\savev2_valuernnnetwork_valuernnnetwork_dynamic_unroll_1_lstm_cell_3_bias_read_readvariableop9savev2_valuernnnetwork_dense_4_kernel_read_readvariableop7savev2_valuernnnetwork_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*
_input_shapesё
ю: : :	Ш:Ш:	Шd:d:	d:
::
:
:::	::	Ш:Ш:	Шd:d:	d:
::
:
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	Ш:!

_output_shapes	
:Ш:%!

_output_shapes
:	Шd: 

_output_shapes
:d:%!

_output_shapes
:	d:&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:&
"
 
_output_shapes
:
:!

_output_shapes	
:: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	Ш:!

_output_shapes	
:Ш:%!

_output_shapes
:	Шd: 

_output_shapes
:d:%!

_output_shapes
:	d:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: "БL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ї
actionь
4

0/discount&
action_0/discount:0џџџџџџџџџ
>
0/observation-
action_0/observation:0џџџџџџџџџ
0
0/reward$
action_0/reward:0џџџџџџџџџ
6
0/step_type'
action_0/step_type:0џџџџџџџџџ
W
1/actor_network_state/0/0:
"action_1/actor_network_state/0/0:0џџџџџџџџџ
W
1/actor_network_state/0/1:
"action_1/actor_network_state/0/1:0џџџџџџџџџ
W
1/actor_network_state/1/0:
"action_1/actor_network_state/1/0:0џџџџџџџџџ
W
1/actor_network_state/1/1:
"action_1/actor_network_state/1/1:0џџџџџџџџџ:
action0
StatefulPartitionedCall:0џџџџџџџџџR
state/actor_network_state/0/01
StatefulPartitionedCall:1џџџџџџџџџR
state/actor_network_state/0/11
StatefulPartitionedCall:2џџџџџџџџџR
state/actor_network_state/1/01
StatefulPartitionedCall:3џџџџџџџџџR
state/actor_network_state/1/11
StatefulPartitionedCall:4џџџџџџџџџtensorflow/serving/predict*ў
get_initial_stateш
2

batch_size$
get_initial_state_batch_size:0 D
actor_network_state/0/0)
PartitionedCall:0џџџџџџџџџD
actor_network_state/0/1)
PartitionedCall:1џџџџџџџџџD
actor_network_state/1/0)
PartitionedCall:2џџџџџџџџџD
actor_network_state/1/1)
PartitionedCall:3џџџџџџџџџtensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:и
щ
policy_state_spec

train_step
metadata
model_variables
_all_assets

signatures
Ѓaction
Єdistribution
Ѕget_initial_state
Іget_metadata
Їget_train_step"
_generic_user_object
9
actor_network_state"
trackable_dict_wrapper
:	 (2global_step
 "
trackable_dict_wrapper
п
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24"
trackable_tuple_wrapper
5
!0
"1
#2"
trackable_list_wrapper
d
Јaction
Љget_initial_state
Њget_train_step
Ћget_metadata"
signature_map
/
$0
%1"
trackable_tuple_wrapper
g:e	Ш2TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel
a:_Ш2RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias
i:g	Шd2VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/kernel
b:`d2TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense_1/bias
j:h	d2WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/kernel
u:s
2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/recurrent_kernel
d:b2UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/bias
m:k
2YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/kernel
w:u
2cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/recurrent_kernel
f:d2WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell_1/bias
Q:O2CActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias
d:b	2QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel
]:[2OActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias
Q:O	Ш2>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/kernel
K:IШ2<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_2/bias
Q:O	Шd2>ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/kernel
J:Hd2<ValueRnnNetwork/ValueRnnNetwork/EncodingNetwork/dense_3/bias
V:T	d2CValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/kernel
a:_
2MValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/recurrent_kernel
P:N2AValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_2/bias
W:U
2CValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/kernel
a:_
2MValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/recurrent_kernel
P:N2AValueRnnNetwork/ValueRnnNetwork/dynamic_unroll_1/lstm_cell_3/bias
1:/	2ValueRnnNetwork/dense_4/kernel
*:(2ValueRnnNetwork/dense_4/bias
1
&ref
&1"
trackable_tuple_wrapper
1
'ref
'1"
trackable_tuple_wrapper
1
(ref
(1"
trackable_tuple_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
9
actor_network_state"
trackable_dict_wrapper
3
	&state
&1"
trackable_tuple_wrapper
u
)_actor_network
&_policy_state_spec
*_policy_step_spec
+_value_network"
_generic_user_object

_state_spec
,_lstm_encoder
-_projection_networks
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+Ќ&call_and_return_all_conditional_losses
­__call__"И
_tf_keras_layer{"class_name": "ActorDistributionRnnNetwork", "name": "ActorDistributionRnnNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
3
	&state
&1"
trackable_tuple_wrapper
ё
2_state_spec
3_lstm_encoder
4_postprocessing_layers
5	variables
6regularization_losses
7trainable_variables
8	keras_api
+Ў&call_and_return_all_conditional_losses
Џ__call__" 
_tf_keras_layer{"class_name": "ValueRnnNetwork", "name": "ValueRnnNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}

_state_spec
9_input_encoder
:_lstm_network
;_output_encoder
<	variables
=regularization_losses
>trainable_variables
?	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"А
_tf_keras_layer{"class_name": "LSTMEncodingNetwork", "name": "ActorDistributionRnnNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
щ
@_means_projection_layer
	A_bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"А
_tf_keras_layer{"class_name": "NormalProjectionNetwork", "name": "NormalProjectionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
~
0
	1

2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
	1

2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
А
Fmetrics
.	variables

Glayers
Hnon_trainable_variables
Ilayer_metrics
/regularization_losses
Jlayer_regularization_losses
0trainable_variables
­__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
/
K0
L1"
trackable_tuple_wrapper

2_state_spec
M_input_encoder
N_lstm_network
O_output_encoder
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"Є
_tf_keras_layer{"class_name": "LSTMEncodingNetwork", "name": "ValueRnnNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}


kernel
 bias
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"ь
_tf_keras_layerв{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
v
0
1
2
3
4
5
6
7
8
9
10
 11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
 11"
trackable_list_wrapper
А
Xmetrics
5	variables

Ylayers
Znon_trainable_variables
[layer_metrics
6regularization_losses
\layer_regularization_losses
7trainable_variables
Џ__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
Э
]_postprocessing_layers
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+И&call_and_return_all_conditional_losses
Й__call__" 
_tf_keras_layer{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ё
bcell
c	variables
dregularization_losses
etrainable_variables
f	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"
_tf_keras_layerь{"class_name": "DynamicUnroll", "name": "dynamic_unroll", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dynamic_unroll", "trainable": true, "dtype": "float32", "parallel_iterations": 20, "swap_memory": null, "cell": {"class_name": "StackedRNNCells", "config": {"name": "stacked_rnn_cells", "trainable": true, "dtype": "float32", "cells": [{"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTMCell", "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}]}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 100]}}
 "
trackable_list_wrapper
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
А
gmetrics
<	variables

hlayers
inon_trainable_variables
jlayer_metrics
=regularization_losses
klayer_regularization_losses
>trainable_variables
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
ъ

kernel
bias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"У
_tf_keras_layerЉ{"class_name": "Dense", "name": "means_projection_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "means_projection_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.1, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
х
bias
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+О&call_and_return_all_conditional_losses
П__call__"Ъ
_tf_keras_layerА{"class_name": "BiasLayer", "name": "bias_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bias_layer", "trainable": true, "dtype": "float32", "bias_initializer": {"class_name": "Constant", "config": {"value": -0.8697231582271624}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1]}}
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
А
tmetrics
B	variables

ulayers
vnon_trainable_variables
wlayer_metrics
Cregularization_losses
xlayer_regularization_losses
Dtrainable_variables
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
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
Э
y_postprocessing_layers
z	variables
{regularization_losses
|trainable_variables
}	keras_api
+Р&call_and_return_all_conditional_losses
С__call__" 
_tf_keras_layer{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ќ
~cell
	variables
regularization_losses
trainable_variables
	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"
_tf_keras_layerє{"class_name": "DynamicUnroll", "name": "dynamic_unroll_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dynamic_unroll_1", "trainable": true, "dtype": "float32", "parallel_iterations": 20, "swap_memory": null, "cell": {"class_name": "StackedRNNCells", "config": {"name": "stacked_rnn_cells_1", "trainable": true, "dtype": "float32", "cells": [{"class_name": "LSTMCell", "config": {"name": "lstm_cell_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTMCell", "config": {"name": "lstm_cell_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}]}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 100]}}
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
Е
metrics
P	variables
layers
non_trainable_variables
layer_metrics
Qregularization_losses
 layer_regularization_losses
Rtrainable_variables
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
Е
metrics
T	variables
layers
non_trainable_variables
layer_metrics
Uregularization_losses
 layer_regularization_losses
Vtrainable_variables
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
Е
metrics
^	variables
layers
non_trainable_variables
layer_metrics
_regularization_losses
 layer_regularization_losses
`trainable_variables
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
К

cells
	variables
regularization_losses
trainable_variables
	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"
_tf_keras_layerџ{"class_name": "StackedRNNCells", "name": "stacked_rnn_cells", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stacked_rnn_cells", "trainable": true, "dtype": "float32", "cells": [{"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTMCell", "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}]}}
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
J
0
1
2
3
4
5"
trackable_list_wrapper
Е
metrics
c	variables
layers
non_trainable_variables
layer_metrics
dregularization_losses
 layer_regularization_losses
etrainable_variables
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Е
metrics
l	variables
 layers
Ёnon_trainable_variables
Ђlayer_metrics
mregularization_losses
 Ѓlayer_regularization_losses
ntrainable_variables
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
Е
Єmetrics
p	variables
Ѕlayers
Іnon_trainable_variables
Їlayer_metrics
qregularization_losses
 Јlayer_regularization_losses
rtrainable_variables
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
8
Љ0
Њ1
Ћ2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Е
Ќmetrics
z	variables
­layers
Ўnon_trainable_variables
Џlayer_metrics
{regularization_losses
 Аlayer_regularization_losses
|trainable_variables
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
Р

Бcells
В	variables
Гregularization_losses
Дtrainable_variables
Е	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"
_tf_keras_layer{"class_name": "StackedRNNCells", "name": "stacked_rnn_cells_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stacked_rnn_cells_1", "trainable": true, "dtype": "float32", "cells": [{"class_name": "LSTMCell", "config": {"name": "lstm_cell_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTMCell", "config": {"name": "lstm_cell_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}]}}
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
З
Жmetrics
	variables
Зlayers
Иnon_trainable_variables
Йlayer_metrics
regularization_losses
 Кlayer_regularization_losses
trainable_variables
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
M0
N1"
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
ш
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ш

kernel
	bias
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1]}}
а


kernel
bias
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"Ѕ
_tf_keras_layer{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 200]}}
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ч0
Ш1"
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
J
0
1
2
3
4
5"
trackable_list_wrapper
И
Щmetrics
	variables
Ъlayers
Ыnon_trainable_variables
Ьlayer_metrics
regularization_losses
 Эlayer_regularization_losses
trainable_variables
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
b0"
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
ь
Ю	variables
Яregularization_losses
аtrainable_variables
б	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"з
_tf_keras_layerН{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ь

kernel
bias
в	variables
гregularization_losses
дtrainable_variables
е	keras_api
+а&call_and_return_all_conditional_losses
б__call__"Ё
_tf_keras_layer{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1]}}
а

kernel
bias
ж	variables
зregularization_losses
иtrainable_variables
й	keras_api
+в&call_and_return_all_conditional_losses
г__call__"Ѕ
_tf_keras_layer{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 200]}}
 "
trackable_list_wrapper
8
Љ0
Њ1
Ћ2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
к0
л1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
И
мmetrics
В	variables
нlayers
оnon_trainable_variables
пlayer_metrics
Гregularization_losses
 рlayer_regularization_losses
Дtrainable_variables
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
~0"
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
И
сmetrics
Л	variables
тlayers
уnon_trainable_variables
фlayer_metrics
Мregularization_losses
 хlayer_regularization_losses
Нtrainable_variables
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
И
цmetrics
П	variables
чlayers
шnon_trainable_variables
щlayer_metrics
Рregularization_losses
 ъlayer_regularization_losses
Сtrainable_variables
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
И
ыmetrics
У	variables
ьlayers
эnon_trainable_variables
юlayer_metrics
Фregularization_losses
 яlayer_regularization_losses
Хtrainable_variables
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
Ћ

kernel
recurrent_kernel
bias
№	variables
ёregularization_losses
ђtrainable_variables
ѓ	keras_api
+д&call_and_return_all_conditional_losses
е__call__"ъ
_tf_keras_layerа{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
Џ

kernel
recurrent_kernel
bias
є	variables
ѕregularization_losses
іtrainable_variables
ї	keras_api
+ж&call_and_return_all_conditional_losses
з__call__"ю
_tf_keras_layerд{"class_name": "LSTMCell", "name": "lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
0
Ч0
Ш1"
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
И
јmetrics
Ю	variables
љlayers
њnon_trainable_variables
ћlayer_metrics
Яregularization_losses
 ќlayer_regularization_losses
аtrainable_variables
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
И
§metrics
в	variables
ўlayers
џnon_trainable_variables
layer_metrics
гregularization_losses
 layer_regularization_losses
дtrainable_variables
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
И
metrics
ж	variables
layers
non_trainable_variables
layer_metrics
зregularization_losses
 layer_regularization_losses
иtrainable_variables
г__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
Џ

kernel
recurrent_kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+и&call_and_return_all_conditional_losses
й__call__"ю
_tf_keras_layerд{"class_name": "LSTMCell", "name": "lstm_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
Џ

kernel
recurrent_kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+к&call_and_return_all_conditional_losses
л__call__"ю
_tf_keras_layerд{"class_name": "LSTMCell", "name": "lstm_cell_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
0
к0
л1"
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
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
И
metrics
№	variables
layers
non_trainable_variables
layer_metrics
ёregularization_losses
 layer_regularization_losses
ђtrainable_variables
е__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
И
metrics
є	variables
layers
non_trainable_variables
layer_metrics
ѕregularization_losses
 layer_regularization_losses
іtrainable_variables
з__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
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
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
И
metrics
	variables
layers
non_trainable_variables
layer_metrics
regularization_losses
 layer_regularization_losses
trainable_variables
й__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
И
metrics
	variables
layers
 non_trainable_variables
Ёlayer_metrics
regularization_losses
 Ђlayer_regularization_losses
trainable_variables
л__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
2
__inference_action_117169
__inference_action_117495Ч
ОВК
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
defaultsЂ	
Ђ 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
"__inference_distribution_fn_117804Ћ
ЄВ 
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
annotationsЊ *
 
в2Я
$__inference_get_initial_state_117832І
В
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
annotationsЊ *
 
ЏBЌ
__inference_<lambda>_1192"
В
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
annotationsЊ *
 
ЏBЌ
__inference_<lambda>_1189"
В
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
annotationsЊ *
 
оBл
$__inference_signature_wrapper_146237
0/discount0/observation0/reward0/step_type1/actor_network_state/0/01/actor_network_state/0/11/actor_network_state/1/01/actor_network_state/1/1"
В
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
annotationsЊ *
 
ЮBЫ
$__inference_signature_wrapper_146250
batch_size"
В
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
annotationsЊ *
 
РBН
$__inference_signature_wrapper_146258"
В
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
annotationsЊ *
 
РBН
$__inference_signature_wrapper_146262"
В
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
annotationsЊ *
 
т2пм
гВЯ
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2бЮ
ХВС
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
kwonlydefaultsЊ 
annotationsЊ *
 
д2бЮ
ХВС
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
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
ц2ур
зВг
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
с2ол
вВЮ
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
kwonlydefaultsЊ 
annotationsЊ *
 
с2ол
вВЮ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
ц2ур
зВг
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
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
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
с2ол
вВЮ
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
kwonlydefaultsЊ 
annotationsЊ *
 
с2ол
вВЮ
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
kwonlydefaultsЊ 
annotationsЊ *
 
л2ие
ЬВШ
FullArgSpec@
args85
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
л2ие
ЬВШ
FullArgSpec@
args85
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
л2ие
ЬВШ
FullArgSpec@
args85
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
л2ие
ЬВШ
FullArgSpec@
args85
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
Ф2СО
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
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
kwonlydefaultsЊ 
annotationsЊ *
 8
__inference_<lambda>_1189Ђ

Ђ 
Њ " 	1
__inference_<lambda>_1192Ђ

Ђ 
Њ "Њ л
__inference_action_117169Н	
НЂЙ
БЂ­
юВъ
TimeStep6
	step_type)&
time_step/step_typeџџџџџџџџџ0
reward&#
time_step/rewardџџџџџџџџџ4
discount(%
time_step/discountџџџџџџџџџ>
observation/,
time_step/observationџџџџџџџџџ
ЕЊБ
Ў
actor_network_stateЂ

?<
$policy_state/actor_network_state/0/0џџџџџџџџџ
?<
$policy_state/actor_network_state/0/1џџџџџџџџџ

?<
$policy_state/actor_network_state/1/0џџџџџџџџџ
?<
$policy_state/actor_network_state/1/1џџџџџџџџџ

 
Њ "ыВч

PolicyStep*
action 
actionџџџџџџџџџ
stateЊ

actor_network_stateіЂђ
wt
85
state/actor_network_state/0/0џџџџџџџџџ
85
state/actor_network_state/0/1џџџџџџџџџ
wt
85
state/actor_network_state/1/0џџџџџџџџџ
85
state/actor_network_state/1/1џџџџџџџџџ
infoЂ ћ
__inference_action_117495н	
нЂй
бЂЭ
ЦВТ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ4
observation%"
observationџџџџџџџџџ
§Њљ
і
actor_network_stateоЂк
kh
2/
actor_network_state/0/0џџџџџџџџџ
2/
actor_network_state/0/1џџџџџџџџџ
kh
2/
actor_network_state/1/0џџџџџџџџџ
2/
actor_network_state/1/1џџџџџџџџџ

 
Њ "ыВч

PolicyStep*
action 
actionџџџџџџџџџ
stateЊ

actor_network_stateіЂђ
wt
85
state/actor_network_state/0/0џџџџџџџџџ
85
state/actor_network_state/0/1џџџџџџџџџ
wt
85
state/actor_network_state/1/0џџџџџџџџџ
85
state/actor_network_state/1/1џџџџџџџџџ
infoЂ ў
"__inference_distribution_fn_117804з	
йЂе
ЭЂЩ
ЦВТ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ4
observation%"
observationџџџџџџџџџ
§Њљ
і
actor_network_stateоЂк
kh
2/
actor_network_state/0/0џџџџџџџџџ
2/
actor_network_state/0/1џџџџџџџџџ
kh
2/
actor_network_state/1/0џџџџџџџџџ
2/
actor_network_state/1/1џџџџџџџџџ
Њ "щВх

PolicyStepЇ
action№сУЂ~
`
CЂ@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
.Њ+
)
loc"
Identityџџџџџџџџџ
Њ _TFPTypeSpec
stateЊ

actor_network_stateіЂђ
wt
85
state/actor_network_state/0/0џџџџџџџџџ
85
state/actor_network_state/0/1џџџџџџџџџ
wt
85
state/actor_network_state/1/0џџџџџџџџџ
85
state/actor_network_state/1/1џџџџџџџџџ
infoЂ Э
$__inference_get_initial_state_117832Є"Ђ
Ђ


batch_size 
Њ "§Њљ
і
actor_network_stateоЂк
kh
2/
actor_network_state/0/0џџџџџџџџџ
2/
actor_network_state/0/1џџџџџџџџџ
kh
2/
actor_network_state/1/0џџџџџџџџџ
2/
actor_network_state/1/1џџџџџџџџџў
$__inference_signature_wrapper_146237е	
ЄЂ 
Ђ 
Њ
.

0/discount 

0/discountџџџџџџџџџ
8
0/observation'$
0/observationџџџџџџџџџ
*
0/reward
0/rewardџџџџџџџџџ
0
0/step_type!
0/step_typeџџџџџџџџџ
Q
1/actor_network_state/0/041
1/actor_network_state/0/0џџџџџџџџџ
Q
1/actor_network_state/0/141
1/actor_network_state/0/1џџџџџџџџџ
Q
1/actor_network_state/1/041
1/actor_network_state/1/0џџџџџџџџџ
Q
1/actor_network_state/1/141
1/actor_network_state/1/1џџџџџџџџџ"Њ
*
action 
actionџџџџџџџџџ
Y
state/actor_network_state/0/085
state/actor_network_state/0/0џџџџџџџџџ
Y
state/actor_network_state/0/185
state/actor_network_state/0/1џџџџџџџџџ
Y
state/actor_network_state/1/085
state/actor_network_state/1/0џџџџџџџџџ
Y
state/actor_network_state/1/185
state/actor_network_state/1/1џџџџџџџџџ
$__inference_signature_wrapper_146250ѕ0Ђ-
Ђ 
&Њ#
!

batch_size

batch_size "РЊМ
M
actor_network_state/0/02/
actor_network_state/0/0џџџџџџџџџ
M
actor_network_state/0/12/
actor_network_state/0/1џџџџџџџџџ
M
actor_network_state/1/02/
actor_network_state/1/0џџџџџџџџџ
M
actor_network_state/1/12/
actor_network_state/1/1џџџџџџџџџX
$__inference_signature_wrapper_1462580Ђ

Ђ 
Њ "Њ

int64
int64 	<
$__inference_signature_wrapper_146262Ђ

Ђ 
Њ "Њ 