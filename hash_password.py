import streamlit_authenticator as stauth

# Create hasher instance
hasher = stauth.Hasher()

# Hash David's password
password = "Purrgatofunworks"
hashed_password = hasher.hash(password)

print(f"Password: {password}")
print(f"Hashed password: {hashed_password}")
print(f"Hash length: {len(hashed_password)}")