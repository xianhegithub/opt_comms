import komm

# for polynomial of degree 8
field = komm.FiniteBifield(12)
# x represents the polynomial 1+x^7
x = field(0b10000001)
# y represents the polynomial 1+x+x^4
y = field(0b00010011)

print(x*y)
# the result is 0b100110010011, representing the polynomial 1+x+x^4+x^7+x^8+x^11
