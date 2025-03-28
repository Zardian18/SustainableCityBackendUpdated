optimized_coords = []
    # with app.test_client() as client:
    #     response = client.get(f'/optimized_route?start={start}&end={end}')
    #     optimized_data = response.json

    #     try:
    #         # Extract coordinates from each section
    #         for section in optimized_data['routes'][0]['sections']:
    #             if 'polyline' in section:
    #                 # Decode polyline (HERE API uses encoded polylines)
    #                 polyline_decoded = polyline.decode(section['polyline'])
    #                 optimized_coords.extend(polyline_decoded)

    #         # Remove duplicates while preserving order
    #         seen = set()
    #         optimized_coords = [x for x in optimized_coords if tuple(x) not in seen and not seen.add(tuple(x))]

    #     except (KeyError, IndexError) as e:
    #         print(f"Error parsing optimized route: {e}")
