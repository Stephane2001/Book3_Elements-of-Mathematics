# <center>Chapter_1</center>

**需要用到的函数**
```python
"""

float()             将输入转化为浮点数
input()             函数接受一个标准输入数据，返回为字符串str类型
int()               将输入转化为整数
is_integer()        判断是否为整数
lambda()            构造匿名函数
len()               返回序列或者数据帧的数据数量
math.e              math库中的欧拉数
math.pi             math库中的圆周率
math.sqrt(2)        math库计算2的平方根
mpmath.e            mpmath库中的欧拉数
mpmath.pi           mpmath库中的圆周率
mpmath.sqrt(2)      mpmath库计算2的平方根
numpy.add()         向量或矩阵加法
numpy.array()       构造数组、向量或矩阵
numpy.cusum()       计算累计求和
numpy.linspace()    在指定的间隔内，返回固定步长数组
numpy.matrix()      构造二维矩阵
print()             在console打印
range()             返回的是一个可迭代对象
zip(*)              将可迭代对象作为参数，让对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
                    *代表解包，返回的每一个都是元组类型，而非是原来的数据类型

"""
```


```python
# 需要的库文件

import math
from mpmath import mp
import numpy as np
```

## <center>1.2 数字分类： 从复数到自然数</center>
### 1.2.1 复数
复数包括实数和虚数。复数集为C。
复数的具体形式为：${a + bi}$
- 其中a和b是实数，i是虚数单位，有${i^2 = -1}$

### 1.2.2 实数
实数集记号为R。实数包括有理数和无理数。实数集合可以用实数轴来展示。
- 有理数：集合用Q表示，有理数可以表示为两个整数的商，比如$\frac{a}{b}$
    - 其中a为分子，b为分母，并且分母不为零。
    - 有理数可以表达为有限小数或无限循环小数。
- 无理数：也叫无限不循环小数。
    - 有很多重要数值都是无理数，例如${\pi}$、${\sqrt{2}}$、${e}$等。


```python
"""打印pi、sqrt(2)、e的精确值"""

import math

print(math.pi)
print(math.e)
print(math.sqrt(2))
```

    3.141592653589793
    2.718281828459045
    1.4142135623730951
    


```python
"""打印pi、sqrt(2)、e的精确值（精确到小数点后1000位）"""

from mpmath import mp

mp.dps = 1000 + 1  # 精度为1000 + 1

print("\nprint 1000 digits of pi behind decimal point: \n")
print(mp.pi)

print("\nprint 1000 digits of e behind decimal point: \n")
print(mp.e)

print("\nprint 1000 digits of sqrt(2) behind decimal point: \n")
print(mp.sqrt(2))
```

    
    print 1000 digits of pi behind decimal point: 
    
    3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989
    
    print 1000 digits of e behind decimal point: 
    
    2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274274663919320030599218174135966290435729003342952605956307381323286279434907632338298807531952510190115738341879307021540891499348841675092447614606680822648001684774118537423454424371075390777449920695517027618386062613313845830007520449338265602976067371132007093287091274437470472306969772093101416928368190255151086574637721112523897844250569536967707854499699679468644549059879316368892300987931277361782154249992295763514822082698951936680331825288693984964651058209392398294887933203625094431173012381970684161403970198376793206832823764648042953118023287825098194558153017567173613320698112509961818815930416903515988885193458072738667385894228792284998920868058257492796104841984443634632449684875602336248270419786232090021609902353043699418491463140934317381436405462531520961836908887070167683964243781405927145635490613031072085103837505101157477041718986106873969655212671546889570350354
    
    print 1000 digits of sqrt(2) behind decimal point: 
    
    1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727350138462309122970249248360558507372126441214970999358314132226659275055927557999505011527820605714701095599716059702745345968620147285174186408891986095523292304843087143214508397626036279952514079896872533965463318088296406206152583523950547457502877599617298355752203375318570113543746034084988471603868999706990048150305440277903164542478230684929369186215805784631115966687130130156185689872372352885092648612494977154218334204285686060146824720771435854874155657069677653720226485447015858801620758474922657226002085584466521458398893944370926591800311388246468157082630100594858704003186480342194897278290641045072636881313739855256117322040245091227700226941127573627280495738108967504018369868368450725799364729060762996941380475654823728997180326802474420629269124859052181004459842150591120249441341728531478105803603371077309182869314710171111683916581726889419758716582152128229518488472
    

### 1.2.3 整数
整数包括正整数、负整数和零。
- 正整数**大于零**， 负整数**小于零**。整数集用Z表示。
- 整数的重要性质
    - 整数**相加**、**相减**或**相乘**的结果依然是整数。
    - 奇偶性
        - 能被2整数的整数称为**偶数**；
        - 否则为**奇数**。


```python
"""判断数字的奇偶性"""

number = float(input("Enter a number: "))

if number.is_integer():
    if (number % 2) == 0:
        print("The number {0} is an even.".format(int(number)))
    else:
        print("The number {0} is an odd.".format(int(number)))

else:
    print("The number {0} is not a integer.".format(number))
```

    Enter a number:  1
    

    The number 1 is an odd.
    

### 1.2.4 自然数
自然数有时候指的是正整数，有时候指非负整数，0是否属于自然数，还存在争议。
    
### 1.2.5 英文对照
下表展示了本节出现的名词的中英文对照，部分有相应的举例。
    
|英文表达|汉语表达|举例|
|:---:  |:---:   |:---:|
|**complex number**|复数|3+7i|
|**real number**|实数|10|
|**imaginary number**|虚数|6i|
|**imaginary unit**|虚数单位|i|
|**rational number**|有理数|30|
|**irrational number**|无理数|${\pi}$|
|**number line**|数轴|
|**quotient**|商|
|**numerator**|分子|
|**denominator**|分母|
|**decimal separator**|小数点|
|**exponential constant**|自然常数|*e*|
|**natural numbers**|自然数|20|
|**parity**|奇偶性||
|**even**|偶数|2|
|**odd**|奇数|1|
|**the set of complex number**|复数集|*C*|
|**the set of real number**|实数集|*R*|
|**the set of rational number**|有理数集|*Q*|

<center>表1. 中英文名词对照表</center>

    

## <center>1.3 加减： 最基本的数学运算</center>

### 1.3.1 加法
**加法**的运算符为**加号**；
在加法算式中，**等式**的左边为**加数**和**被加数**，等式的右边是**和**。

加法的表达方式有
- 和
- 加
- 增长
- 小计
- 总数






```python
"""加法运算"""

# user input numbers
num1 = input('Enter first number: ')
num2 = input('Enter second number: ')

# add two numbers
sum = float(num1) + float(num2)

# display the computation
print('The sum of {0} and {1} is {2}'.format(num1, num2, sum))
```

    Enter first number:  12.2
    Enter second number:  10.9
    

    The sum of 12.2 and 10.9 is 23.1
    

### 1.3.2 累计求和

累计求和得到的结果不是一个总和，而是从左向右每加一个数值，得到的分步结果。例如，自然数1到10的累计求和结果为：
- 1, 3, 6, 10, 15, 21, 28, 36, 45, 55




```python
"""对1到10这十个自然数进行累加"""

a_i = np.linspace(1,10,10)
print(a_i)

a_i_cumsum = np.cumsum(a_i)
print(a_i_cumsum)
```

    [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
    [ 1.  3.  6. 10. 15. 21. 28. 36. 45. 55.]
    

### 1.3.3 减法
**减法**是**加法的逆运算**，运算符为**减号**。减法运算的过程为，**被减数**减去**减数**得到**差**。

减法的表达方式有：
- 减
- 少
- 差
- 减少
- 拿走
- 扣除


```python
"""减法运算"""

num1 = 5
num2 = 3

# add two numbers
diff = num1 - num2

# display the computation
print('The difference of {0} and {1} is {2}'.format(num1, num2, diff))
```

    The difference of 5 and 3 is 2
    

### 1.3.4 相反数

求**相反数**的的过程是**改变符号**，这个操作被称为**变号**。例如。5的相反数是-5。

### 1.3.5 中英文对照

|数学表达|英文表达|
|:---:|:---|
|1+1=2|One plus one equals two.<br>The sum of one and one is two.<br>If you add one to one, you get two|
|2+3=5|Two plus three equals five.<br>Two plus three is equal to five.<br>Three added to two makes five.<br>If you add two to three, you get five.|

<center>表2. 加法的英文表达</center>

|数学表达|英文表达|
|:---:|:---|
|5-3=2| Five minus three equals two.<br>Five minus three is equal to two.<br>Three substracted from five equals two.<br>If you substract three from five, you get two.<br>If you take three from five, you get two.|
|4-6=-2|Four minus six equals negative two.<br>Four minus is equal to negative two.|
<center>表3. 减法的英文表达</center>

|英文表达|汉语表达|举例|
|:---:|:---:|:---:|
|**addition**|加法||
|**plus sign, plus symbol**|加号|+|
|**equation**|等式|
|**addend, summand**|加数|3 in ***3+5=8***|
|**augend, summand**|被加数|5 in ***3+5=8***|
|**sum**|和|8 in ***3+5=8***|
|**summation**|和||
|**plus**|加||
|**increase**|增长||
|**subtotal**|小计||
|**total**|总数||
|**cumulative sum, cumulative total**|累计求和||
|**subtraction**|减法||
|**inverse operation of addition**|加法的逆运算||
|**minus sign**|减号|-|
|**minuend**|被减数|5 in ***5-2=3***|
|**subtrahend**|减数|2 in ***5-2=3***|
|**difference**|差|3 in ***5-2=3***|
|**minus**|减||
|**less**|少||
|**decrease**|减少||
|**take away**|拿走||
|**deduct**|扣除||
|**inverse number, additive inverse number**|相反数|5 and -5|
|**reverses its sign, sign change**|变号|5 -> -5|

<center>表4. 中英文名词对照表</center>

## <center>1.4 向量，数字排成行、列</center>

### 1.4.1 行向量、列向量
若干数字排成一行或一列，并用中括号括起来，得到的数组叫做**向量**。

排成一行的叫做**行向量**，排成一列的叫做**列向量**。
$$\begin{bmatrix} 1 & 2 & 3\end{bmatrix}_{1 \times 3}$$

$$\begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix}_{3 \times 1}$$

其中，${1 \times 3}$代表 *1行，3列*； ${3 \times 1}$代表 *3行，1列*。

```python
# 在numpy库中，这样定义行向量与列向量

numpy.array([1, 2, 3])  # 定义行向量
numpy.array([1], [2], [3])  # 定义列向量
```
给定如下行向量***a***，***a***有n个元素，元素本身用小写字母表示，如：
$$\textbf{a} = \begin{bmatrix} {a}_{1} & {a}_{2} & \cdots & {a}_{n} \end{bmatrix}$$


### 1.4.2 转置
使用转置符号${}^{T}$。行向量转置得到列向量，列向量转置得到行向量。
例如：
$$\begin{bmatrix} 1 & 2 & 3\end{bmatrix}^{T} = \begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix}$$

$$\begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix}^{T} = \begin{bmatrix} 1 & 2 & 3\end{bmatrix}$$

### 1.4.3 英文对照

|英文表达|汉语表达|举例|
|:---:|:---:|:---:|
|**vector**|向量|$$\begin{bmatrix} 1 & 2 & 3\end{bmatrix}_{1 \times 3}$$<br>$$\begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix}_{3 \times 1}$$|
|**row vector**|行向量|$$\begin{bmatrix} 1 & 2 & 3\end{bmatrix}_{1 \times 3}$$|
|**column vector**|列向量|$$\begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix}_{3 \times 1}$$|
|**transpose**|转置|$$\begin{bmatrix} 1 & 2 & 3\end{bmatrix}^{T} = \begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix}$$<br>$$\begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix}^{T} = \begin{bmatrix} 1 & 2 & 3\end{bmatrix}$$|  
<center>表5. 中英文名词对照表</center>


```python
"""定义行向量和列向量,并得到各自的转置"""

import numpy as np

row = np.array([4, 6, 9])  # 定义行向量
col = np.array([[5], [10], [20]])  # 定义列向量

print(row)
print(col)

row_tr = row.T  # 行向量转置
col_tr = col.T

print(row_tr)
print(col_tr)
```

    [4 6 9]
    [[ 5]
     [10]
     [20]]
    [4 6 9]
    [[ 5 10 20]]
    

## <center>1.5 矩阵：数字排列成长方形</center>  
### 1.5.1 矩阵
**矩阵**将一系列数字以长方形方式排列，比如：
$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}_{2 \times 3}$$  
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \\ \end{bmatrix}_{3 \times 2}$$  
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix}_{2 \times 2}$$  
这三个矩阵的形状分别为 *2行3列*，*3行2列*，*2行2列*。  

一个一般性的矩阵可以这样来表示:  
$$\textbf{X}_{n \times D}=\begin{bmatrix}{x}_{1,1} & {x}_{1,2} & \cdots & {x}_{1,D} \\ {x}_{2,1} & {x}_{2,2} & \cdots & {x}_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ {x}_{n,1} & {x}_{n,2} & \cdots & {x}_{n,D} \\\end{bmatrix}$$  
这个矩阵的形状是${n \times D}$，${x}_{i,j}$被称为矩阵的元素，可以说${x}_{i,j}$出现在第i行，第j列。  

### 1.5.2 英文对照
|数学表达|英文表达|
|:---|:---|
|$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix}$$|Two by two matrix, first row one two, second rwo three four|
|$$\begin{bmatrix}{x}_{1,1} & {x}_{1,2} & \cdots & {x}_{1,n} \\ {x}_{2,1} & {x}_{2,2} & \cdots & {x}_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ {x}_{m,1} & {x}_{m,2} & \cdots & {x}_{m,m} \\\end{bmatrix}$$|m by n matrix,<br>first row a sub one one, a sub one two, dot dot dot, a sub one n<br>second row a sub two one, a sub two two, dot dot dot, a sub two n<br>dot dot dot<br>last row a sub m one, a sub m two, dot dot dot, a sub m n.|
|$${a}_{i,j}$$|Lowercase(small) a sub i comma j|
|$${a}_{i,j+1}$$|Lowercase(small) a sub i comma j plus one|
|$${a}_{i,j-1}$$|Lowercase(small) a sub i comma j minus one|  
<center>表6. 数学英文对照表</center>


```python
"""
定义一个矩阵
提取矩阵的某一列、某两列
提取矩阵的某一行
提取矩阵的某一个位置的具体值
"""

import numpy as np

A = np.array([[1, 2, 6],
              [9, 8, 3]])  # 定义一个3*2的矩阵

A_first_column = A[:, 0]  # 提取第一列
A_first_col_V2 = A[:,[0]]  # 提取第一列，并将其作为列向量
A_first_sencond_column = A[:, [0, 1]]  # 提取第二列

A_first_row = A[0, :]  # 提取第一行
A_element = A[[0], [2]]  # 提取第一行第三列元素

print(A_first_column)
print(A_first_col_V2)
print(A_first_sencond_column)
print(A_first_row)
print(A_element)

```

    [1 9]
    [[1]
     [9]]
    [[1 2]
     [9 8]]
    [1 2 6]
    [6]
    

## <center>1.6 矩阵：一排列向量，或一组行向量</center>  

### 1.6.1 矩阵的组成
矩阵可以看作为若干个列向量左右排列或若干行向量上下叠放。例如：  
$$
\begin{bmatrix} 
1 & 2 & 3 \\ 4 & 5 & 6 
\end{bmatrix}_{2 \times 3} = 
\begin{bmatrix}
\begin{bmatrix}
1 \\ 4 
\end{bmatrix} & 
\begin{bmatrix}
2 \\ 5 
\end{bmatrix} & 
\begin{bmatrix} 
3 \\ 6 
\end{bmatrix}
\end{bmatrix} = 
\begin{bmatrix}
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix} \\
\begin{bmatrix}
4 & 5 & 6
\end{bmatrix} \\ 
\end{bmatrix}
$$  
一般来说，形状为${n \ times D}$的矩阵***X***，可以写成D个左右排列的列向量；也可以写成n个行向量上下叠放。  
$$\textbf{X}_{n \times D} = \begin{bmatrix}{x}_{1} & {x}_{2} & \cdots & {x}_{D}\end{bmatrix}$$
$$\textbf{X}_{n \times D} = \begin{bmatrix}{x}^{(1)} \\ {x}^{(2)} \\ \vdots \\ {x}^{(n)}\end{bmatrix} $$

### 1.6.2 矩阵转置

矩阵转置是指将矩阵的行列互换得到的新矩阵，例如：
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \\ \end{bmatrix}^{\rm T}_{3 \times 2} = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6\end{bmatrix}_{2 \times 3}$$

如果从矩阵中行/列向量的角度来看，则对**矩阵A**转置的结果为：  
$$\textbf{A}^{\rm T} = \begin{bmatrix}{a}_{1} & {a}_{2} & {a}_{3}\end{bmatrix}^{\rm T} = \begin{bmatrix}{a}^{\rm T}_{1} \\ {a}^{\rm T}_{2} \\ {a}^{\rm T}_{3} \\\end{bmatrix}$$
$$\textbf{A}^{\rm T} = \begin{bmatrix}{a}^{(1)} \\ {a}^{(2)} \\ {a}^{3}\end{bmatrix}^{\rm T} = \begin{bmatrix}{a}^{\rm (1)T} & {a}^{\rm (2)T} & {a}^{\rm (3)T}\end{bmatrix}$$

## <center>1.7矩阵形状：每种形状都有特殊性质和用途</center>

1. 行向量和列向量都是特殊的矩阵;
    1. 如果列向量的元素都为1，一般记作$\textbf{1}$，称为全1列向量，简称为**全1向量**;  
    2. 如果列向量的元素都为0，记作$\textbf{0}$，称为**0向量**。
2. 行数和列数相等的矩阵为**方阵**;
3. **对角矩阵**是除主对角线外的元素皆为0的方阵;
    -  **单位矩阵**是元素均为1的对角矩阵,记为 $\textbf{I}$;
4. **对称矩阵**是元素相对与主对角线轴对称的方阵;
5. **零矩阵**指所有元素皆为0的方阵，记为$\textbf{O}$.

|英文表达|汉语表达|举例|
|:---|:---|:---|
|**all-ones vector**|全1向量|$$\begin{bmatrix}1 \\ 1 \\ \vdots \\ 1\end{bmatrix}$$|
|**zero vector**|零向量|$$\begin{bmatrix}0 \\ 0 \\ \vdots \\ 0\end{bmatrix}$$|
|**square matrix**|方阵|$$\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$$|
|**diagonal matrix**|对角矩阵|$$\begin{bmatrix}5 & & \\ & 10 & \\ & & 20\end{bmatrix}$$|
|**identity matrix**|单位矩阵|$$\begin{bmatrix}1 & & \\ & 1 & \\ & & 1\end{bmatrix}$$|
|**symmetric matrix**|对称矩阵|$$\begin{bmatrix}1 & 2 & 9 \\ 2 & 1 & 5 \\ 9 & 5 & 1\end{bmatrix}$$|
|**null matrix**|0矩阵|$$\begin{bmatrix}0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0\end{bmatrix}$$|
<center>表7. 中英文名词对照表</center>

## <center>1.8 矩阵加减：形状形同，对应位置，批量加减</center>

### 1.8.1 向量加减
行向量与列向量也是特殊的矩阵，它们的加减法为：对应元素相加减，结果仍为行/列向量。
$$\begin{bmatrix}1 & 2 & 3\end{bmatrix} + \begin{bmatrix} 2 & 4 & 6 \end{bmatrix} = \begin{bmatrix}3 & 6 & 9\end{bmatrix}$$  
$$\begin{bmatrix}1 & 2 & 3\end{bmatrix} - \begin{bmatrix} 2 & 4 & 6 \end{bmatrix} = \begin{bmatrix}-1 & -2 & -3\end{bmatrix}$$  
$$\begin{bmatrix}1 \\ 2 \\ 3\end{bmatrix} + \begin{bmatrix} 2 \\ 4 \\ 6 \end{bmatrix} = \begin{bmatrix}3 \\ 6 \\ 9\end{bmatrix}$$  
$$\begin{bmatrix}1 \\ 2 \\ 3\end{bmatrix} - \begin{bmatrix} 2 \\ 4 \\ 6 \end{bmatrix} = \begin{bmatrix}-1 \\ -2 \\ -3\end{bmatrix}$$


```python
"""向量的加减"""

import numpy as np

list1=[1, 2, 3]
list2=[4, 5, 6]
print([x + y for x, y in zip(list1, list2)])

print(list(map(lambda x,y:x+y, list1, list2)))

x = np.array(list1)
y = np.array(list2)
print(x+y)

print(np.add(list1,list2))
```

    [5, 7, 9]
    [5, 7, 9]
    [5 7 9]
    [5 7 9]
    

### 1.8.2 矩阵加减  
形状相同的两个矩阵相加的结果也是矩阵。对应元素相加减，形状不变。例如：  
$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}_{2 \times 3} + \begin{bmatrix} 2 & 4 & 6 \\ 8 & 10 & 12\end{bmatrix}_{2 \times 3} = \begin{bmatrix} {1+2} & {2+4} & {3+6} \\ {4+8} & {5+10} & {6+12}\end{bmatrix}_{2 \times 3} = \begin{bmatrix} 3 & 6 & 9 \\ 12 & 15 & 18\end{bmatrix}_{2 \times 3}$$   
$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}_{2 \times 3} - \begin{bmatrix} 2 & 4 & 6 \\ 8 & 10 & 12\end{bmatrix}_{2 \times 3} = \begin{bmatrix} {1-2} & {2-4} & {3-6} \\ {4-8} & {5-10} & {6-12}\end{bmatrix}_{2 \times 3} = \begin{bmatrix} -1 & -2 & -3 \\ -4 & -5 & -6\end{bmatrix}_{2 \times 3}$$  



```python
"""矩阵加减法"""

import numpy as np

array_1 = [[1, 2, 3], [4, 5, 6]]
array_2 = [[2, 4, 6], [8, 10, 12]]

array_1 = np.array(array_1)
array_2 = np.array(array_2)

sum = np.add(array_1, array_2)
diff = np.subtract(array_1, array_2)

print(sum)
print(diff)
```

    [[ 3  6  9]
     [12 15 18]]
    [[-1 -2 -3]
     [-4 -5 -6]]
    


```python

```
