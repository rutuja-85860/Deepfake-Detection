import qrcode

url = " https://angele-unsibilant-separately.ngrok-free.dev"

qr = qrcode.QRCode(border=2)
qr.add_data(url)
qr.make()

qr.print_ascii(invert=True)

print("\nScan this QR to open the project:")
print(url)