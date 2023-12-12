from source.links import link

link_name = "FEniCS project"
user_text = "FEniCS project"
url = "https://fenicsproject.org"

link.xref_links.update({link_name: (user_text, url)})

print('loaded fenics link')