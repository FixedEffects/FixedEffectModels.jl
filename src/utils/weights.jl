##############################################################################
##
## Weight
## 
##############################################################################
#  remove observations with missing or negative weights
isnaorneg(a::AbstractVector) = BitArray(!ismissing(x) & (x > 0) for x in a)
