from parsers import parse_phone, parse_company, load_legal_suffixes

print(parse_phone("+91 98765-43210"))        # expect ("India", "9876543210") with our map
print(parse_phone("044-12345678"))           # expect ("", "04412345678")

terms = load_legal_suffixes()
print(parse_company("Tresata Pvt Ltd", terms))
print(parse_company("Enno Roggemann GmbH & Co. KG", terms))
print(parse_company("First National Bank", terms))
