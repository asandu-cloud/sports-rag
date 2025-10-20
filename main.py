# Using Football API

import requests

url = "https://v3.football.api-sports.io/leagues"
headers = {'x-apisports-key': 'API-FOOTBALL-KEY'}

response = requests.get(url, headers = headers)
data = response.json()

# Print the first few leagues
for league in data['response'][:10]:
    print(
        f"{league['league']['name']} - {league['country']['name']} - ID: {league['league']['id']}"
    )