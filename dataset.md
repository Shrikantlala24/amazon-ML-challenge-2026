--- Inspecting Train Files ---
train_source1.tsv: 2206821 rows | Header: ['entity_id', 'business_name', 'business_address', 'country'] | Countries: {'US': 1323633, 'India': 883188}
train_source2.tsv: 5034616 rows | Header: ['entity_id', 'business_name', 'business_address', 'country'] | Countries: {'India': 2017799, 'US': 3016817}
train_source3.tsv: 5285603 rows | Header: ['entity_id', 'business_name', 'business_address', 'country'] | Countries: {'US': 3170056, 'India': 2115547}

--- Inspecting Ground Truth ---
Total S1 entities in GT: 2206821
Singletons: 123247 (5.58%)
Entities with matches: 2083574 (94.42%)
Total S2 matches linked: 3693619
Total S3 matches linked: 3944746
Match distribution (counts of matches per S1, top 10): [(3, 530841), (4, 484115), (2, 375212), (5, 321957), (6, 164868), (0, 123247), (1, 119157), (7, 63968), (8, 18680), (9, 4205)]

--- Inspecting Test Files ---
test_source1.tsv: 1732544 rows | Header: ['entity_id', 'business_name', 'business_address', 'country'] | Countries: {'US': 663106, 'France': 259452, 'India': 809986}
test_source2.tsv: 4887273 rows | Header: ['entity_id', 'business_name', 'business_address', 'country'] | Countries: {'India': 2312565, 'France': 703378, 'US': 1871330}
test_source3.tsv: 5082316 rows | Header: ['entity_id', 'business_name', 'business_address', 'country'] | Countries: {'India': 2405000, 'France': 731615, 'US': 1945701}