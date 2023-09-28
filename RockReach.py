import pandas as pd
import rocketreach

# Create a Gateway object with your API key
rr = rocketreach.Gateway(api_key='10dd188k84b4cfe6caec51757284cced9bd22d6e')

# Search for a person by name and current employer
search_params = {
    "current_employer": "Quanser"
}

s = rr.person.search().filter(**search_params)

# Execute the search and get the results
result_ = s.execute()

# Get the 'people' data from the response
people_list = result_.people

# Print the DataFrame
print(people_list)
