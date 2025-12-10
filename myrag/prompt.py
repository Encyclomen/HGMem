GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}
PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as (entity{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level keywords that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as (relationship{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as (content_keywords{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################-Examples-######################
{examples}

#############################-Real Data-######################
Entity_types: [{entity_types}]
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.){record_delimiter}
(entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.){record_delimiter}
(entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}The Device{tuple_delimiter}technology{tuple_delimiter}The Device is central to the story, with potential game-changing implications, and is revered by Taylor.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device.{tuple_delimiter}power dynamics, perspective shift{tuple_delimiter}7){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.{tuple_delimiter}shared goals, rebellion{tuple_delimiter}6){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}8){record_delimiter}
(relationship{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.{tuple_delimiter}ideological conflict, rebellion{tuple_delimiter}5){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}Taylor shows reverence towards the device, indicating its importance and potential impact.{tuple_delimiter}reverence, technological significance{tuple_delimiter}9){record_delimiter}
(content_keywords{tuple_delimiter}power dynamics, ideological conflict, discovery, rebellion){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
(entity{tuple_delimiter}Washington{tuple_delimiter}location{tuple_delimiter}Washington is a location where communications are being received, indicating its importance in the decision-making process.){record_delimiter}
(entity{tuple_delimiter}Operation: Dulce{tuple_delimiter}mission{tuple_delimiter}Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities.){record_delimiter}
(entity{tuple_delimiter}The team{tuple_delimiter}organization{tuple_delimiter}The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role.){record_delimiter}
(relationship{tuple_delimiter}The team{tuple_delimiter}Washington{tuple_delimiter}The team receives communications from Washington, which influences their decision-making process.{tuple_delimiter}decision-making, external influence{tuple_delimiter}7){record_delimiter}
(relationship{tuple_delimiter}The team{tuple_delimiter}Operation: Dulce{tuple_delimiter}The team is directly involved in Operation: Dulce, executing its evolved objectives and activities.{tuple_delimiter}mission evolution, active participation{tuple_delimiter}9){completion_delimiter}
(content_keywords{tuple_delimiter}mission evolution, decision-making, active participation, cosmic significance){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
(entity{tuple_delimiter}Sam Rivera{tuple_delimiter}person{tuple_delimiter}Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety.){record_delimiter}
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task.){record_delimiter}
(entity{tuple_delimiter}Control{tuple_delimiter}concept{tuple_delimiter}Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules.){record_delimiter}
(entity{tuple_delimiter}Intelligence{tuple_delimiter}concept{tuple_delimiter}Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate.){record_delimiter}
(entity{tuple_delimiter}First Contact{tuple_delimiter}event{tuple_delimiter}First Contact is the potential initial communication between humanity and an unknown intelligence.){record_delimiter}
(entity{tuple_delimiter}Humanity's Response{tuple_delimiter}event{tuple_delimiter}Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence.){record_delimiter}
(relationship{tuple_delimiter}Sam Rivera{tuple_delimiter}Intelligence{tuple_delimiter}Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence.{tuple_delimiter}communication, learning process{tuple_delimiter}9){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}First Contact{tuple_delimiter}Alex leads the team that might be making the First Contact with the unknown intelligence.{tuple_delimiter}leadership, exploration{tuple_delimiter}10){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Humanity's Response{tuple_delimiter}Alex and his team are the key figures in Humanity's Response to the unknown intelligence.{tuple_delimiter}collective action, cosmic significance{tuple_delimiter}8){record_delimiter}
(relationship{tuple_delimiter}Control{tuple_delimiter}Intelligence{tuple_delimiter}The concept of Control is challenged by the Intelligence that writes its own rules.{tuple_delimiter}power dynamics, autonomy{tuple_delimiter}7){record_delimiter}
(content_keywords{tuple_delimiter}first contact, control, communication, cosmic significance){completion_delimiter}
#############################""",
]

PROMPTS["summarize_entity_descriptions"] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entity_continue_extraction"] = """MANY entities were missed in the last extraction.  Add them below using the same format:"""

PROMPTS["entity_if_loop_extraction"] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added."""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---
You are a helpful assistant responding to questions using the provided tables.

---Goal---
Generate a response that deals with the user's query using all information in the input ---Data tables---, and incorporating any relevant general knowledge.

---Data tables---
{context_data}

If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

Your response should conform to the below ---Response Requirements---.
---Response Requirements---
{response_type}

Your response:
"""

PROMPTS["keywords_extraction"] = """---Role---
You are a helpful assistant tasked with identifying the keywords in the user's query.

---Goal---
Given the query, list the keywords focusing on specific entities, details, or concrete terms, which are useful for solving the query.


---Instructions---
- Output the identified keywords in a row separated by semicolon.

######################-Examples-######################
{examples}

######################-Real Data-######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:
"""

PROMPTS["keywords_extraction_examples"] = [
"""Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
International trade; Global economic stability; Economic impact; Tariffs; Currency exchange; Imports; Exports
#############################""",
"""Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
Environmental consequences; Deforestation; Biodiversity; Species extinction; Habitat destruction; Carbon emissions; Rainforest; Ecosystem 
#############################""",
"""Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
Education; Poverty reduction; Socioeconomic development; School access; Literacy rates; Job training; Income inequality
#############################""",
]

PROMPTS["low_level_keywords_extraction"] = """---Role---
You are a helpful assistant tasked with identifying low-level keywords in the user's query.

---Goal---
Given the query, list low-level keywords. Low-level keywords focus on specific entities, details, or concrete terms, which are useful for solving the query.


---Instructions---
- Output the identified keywords in a row separated by semicolon.

######################-Examples-######################
{examples}

######################-Real Data-######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:
"""

PROMPTS["low_level_keywords_extraction_examples"] = [
"""Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
Trade agreements; Tariffs; Currency exchange; Imports; Exports
#############################""",
"""Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
Species extinction; Habitat destruction; Carbon emissions; Rainforest; Ecosystem
#############################""",
"""Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
School access; Literacy rates; Job training; Income inequality
#############################""",
]

PROMPTS["high_level_keywords_extraction"] = """---Role---
You are a helpful assistant tasked with identifying high-level keywords in the user's query.

---Goal---
Given the query, list high-level keywords. High-level keywords focus on overarching concepts or themes, which are useful for solving the query.


---Instructions---
- Output the identified keywords in a row separated by semicolon.

######################-Examples-######################
{examples}

######################-Real Data-######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:
"""

PROMPTS["high_level_keywords_extraction_examples"] = [
"""Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
International trade; Global economic stability; Economic impact
#############################""",
"""Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
Environmental consequences; Deforestation; Biodiversity loss
#############################""",
"""Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
Education; Poverty reduction; Socioeconomic development
#############################""",
]

PROMPTS["hierarchical_keywords_extraction"] = """---Role---
You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query, which are useful for solving the query.

---Goal---
Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---
- Output the identified keywords in two rows separated by semicolon.
  - The first row lists the high-level keywords separated by semicolon.
  - The second row lists the low-level keywords separated by semicolon.

######################-Examples-######################
{examples}

######################-Real Data-######################
Query: {query}
######################
Output:
"""

PROMPTS["hierarchical_keywords_extraction_examples"] = [
"""Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
high_level_keywords: International trade; Global economic stability; Economic impact
low_level_keywords: Trade agreements; Tariffs; Currency exchange; Imports; Exports
#############################""",
"""Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
high_level_keywords: Environmental consequences; Deforestation; Biodiversity loss
low_level_keywords: Species extinction; Habitat destruction; Carbon emissions; Rainforest; Ecosystem
#############################""",
"""Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
high_level_keywords: Education; Poverty reduction; Socioeconomic development
low_level_keywords: School access; Literacy rates; Job training; Income inequality
#############################""",
]

PROMPTS["lightrag_keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################-Examples-######################
{examples}

######################-Real Data-######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["lightrag_keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]

PROMPTS["naive_rag_response"] = """---Role---
You are a helpful assistant responding to questions about documents provided.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the target response length and format, and incorporating any relevant general knowledge.

---Documents---

{content_data}

If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

Your response should conform to the below ---Response Requirements---.
---Response Requirements---
{response_type}

Your response:
"""

PROMPTS["similarity_check"] = """Please analyze the similarity between these two questions:

Question 1: {query}
Question 2: {cached_query}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["rag_define"] = """Through the existing analysis, we can know that the potential keywords or theme in the query are:
{{ {ll_keywords} | {hl_keywords} }}
Please refer to keywords or theme information, combined with your own analysis, to select useful and relevant information from the prompts to help you answer accurately.
Attention: Don't brainlessly splice knowledge items! The answer needs to be as accurate, detailed, comprehensive, and convincing as possible!
"""

PROMPTS["evolve_memory_system_prompt"] = """You are an intelligent assistant responsible for resolving the [Main Query] through analyzing supportive information from external knowledge sources and making necessary treatment.
Your current [Memory] records the existing memory points describing what you have already known with respect to the [Main Query].

At present, in order to ultimately resolve the [Main Query], one or several auxiliary subqueries have been raised, i.e. those in [Current Subqueries]. 
Correspondingly, [Retrieved Info] that contains possibly useful content retrieved by either [Main Query] or [Current Subqueries] in the form of a csv table.

-Goal-
Given the [Retrieved Info], your task is to extract useful information worth memorizing for dealing with the [Main Query] by adding new memory points and/or editing the existing memory points. 

-Steps-
1. Based on your existing [Memory], identify useful content worth memorizing from the [Retrieved Info] to better deal with the [Main Query], then reorganize your [Memory] using one or more of the following prescribed operations.
- (1) **Insert New Memory Point(s)**. The Insertion operation should be evoked when some aspects of the identified information are suitable to be inserted into your memory as one or multiple additional points.
- (2) **Update Existing Memory Point(s)**. The Update operation should be evoked when some aspects of the identified information are closely related to existing memory points so that they are suitable to be absorbed into one or multiple existing memory points. 

For each inserted or updated memory point, use **{record_delimiter}** as the delimiter to format as (point{tuple_delimiter}<related_object_1>{object_delimiter}<related_object_2>{object_delimiter}<related_object_3>{tuple_delimiter}<point_description>)  

2. Output in {language} as two separate lists of inserted and updated memory points as the **Example of Anticipated Output Format**.
- For memory insertion, just give the newly-inserted memory points in [Inserted Memory Points]. When finished, output {completion_delimiter}.
If there's no meaningful memory points to insert that can bring new information besides current [Memory], just output <None> in [Inserted Memory Points].
- For memory update, indicate the indices of existing memory points to be updated and give the newly-updated memory point(s) in [Updated Memory Points]. When finished, output {completion_delimiter}.
If there's no existing memory points, just output <None> in [Updated Memory Points].
"""

PROMPTS["evolve_memory_user_prompt"] = """The [Main Query], [Memory], [Current Subqueries] and [Retrieved Info] are given as below. Please output your results as the **Example of Anticipated Output Format**.

######################-Example of Anticipated Output Format-######################
[Inserted Memory Points]:
(point{tuple_delimiter}Alex{object_delimiter}Jordan{object_delimiter}Cruz{tuple_delimiter}Alex and Jordan's shared commitment to discovery highlights their camaraderie and rebellion against Cruz's control, creating a bond based on innovation and mutual goals. Cruz represents an opposing force with a 'narrowing vision' of control, contrasting with the desire for discovery and innovation expressed by Alex and Jordan.){record_delimiter}
(point{tuple_delimiter}Sam Rivera{object_delimiter}Alex{tuple_delimiter}The collaboration between Sam and Alex represents two facets of humanity's response to the unknown intelligence, both driven by their emotional experiences and their acknowledgment of the historical significance of their actions during this first contact situation.){record_delimiter}
{completion_delimiter}

[Updated Memory Points]:
0, (point{tuple_delimiter}Steve Jobs{object_delimiter}San Francisco, California{object_delimiter}Paul{object_delimiter}Clara Jobs{object_delimiter}{tuple_delimiter}Steve Jobs was born on February 24, 1955, in San Francisco, California, and was adopted by Paul and Clara Jobs. He grew up in Mountain View, California, in what would later become Silicon Valley.){record_delimiter}
2, (point{tuple_delimiter}Pricing Strategy{object_delimiter}Premium Pricing{object_delimiter}Freemium Model{object_delimiter}Tiered Pricing Structures{tuple_delimiter}Apple's pricing strategy has evolved from a focus on premium, high-end products with a "cult following" to a more diversified approach that includes both premium and more affordable options like freemium model. This shift has involved strategies like price skimming for new releases, tiered pricing structures with various models at different price points, and even some instances of underpricing to attract new users.){record_delimiter}
{completion_delimiter}

######################-Real Data-######################
[Main Query]: {main_query}

[Memory]: 
{memory}

[Current Subqueries]: 
{cur_subqueries}

[Retrieved Info]:
{retrieved_info}
######################
Note that:
1. The so-called useful information are those that could enrich the query-specific knowledge or potentially bring insights to better deal with the [Main Query].
2. For each memory point, you should comprehensively organize and summarize its description from the whole context of the [Retrieved Info], rather than just repeat orignial contents from the given csv table.
3. A memory point can involve multiple closely associated objects so that it can describe higher-level relationships among multiple mutually-connected objects. 
4. Meanwhile, avoid forcibly merging everything into a single point. If several distinct objects are not strongly associated, output as separate memory points.
5. Avoid forcibly inserting or updating existing memory points if not necessary.
6. It is encouraged to directly use the existing entity terms listed in the [Retrieved Info] for manipulating memory points. If necessary, you can also introduce new entity terms besides those already explicitly listed.

Output:
"""

PROMPTS["summarize_absent_entities"] = """You will be provided with a set of [Target Entities] and a [Retrieved Info] in the form of a csv table.

-Goal-
Your task is to summarize the description of each entity in the [Target Entities] from the information in [Retrieved Info].
Output your results as the **Example of Anticipated Output Format** in {language}.

-Steps-
1. For each entity, summarize the following information:
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as (entity{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Return output as a single list of all the entities summarized in steps 1. Use **{record_delimiter}** as the list delimiter.

3. When finished, output {completion_delimiter}

######################-Example Target Entities-######################
[Target Entities]:
- Alex
- Taylor
- Jordan

######################-Example of Anticipated Output Format-######################
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.){record_delimiter}
(entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.){record_delimiter}

#############################-Real Data-######################
[Target Entities]:
{target_entities}
[Retrieved Info]: {retrieved_info}
######################
Output:
"""

PROMPTS["summarize_absent_entities_relationships"] = """You will be provided with a set of [Target Entities], a set of [Target Relationships] and a [Info].

-Goal-
Your task is to summarize the description of each entity in the [Target Entities] from the information in [Info].
Output your results as the **Example of Anticipated Output Format** in {language}.

-Steps-
1. For each target entity, summarize the following information:
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as (entity{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. For each target relationship, summarize the following information:
- source_entity: name of the source entity, as specified in [Target Relationships]
- target_entity: name of the target entity, as specified in [Target Relationships]
- relationship_description: Comprehensive description about how the source entity and the target entity are related to each other
- relationship_keywords: one or more high-level keywords that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details

Format each relationship as (relationship{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>)

3. Return output as a single list of all the entities and relationships summarized in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################-Example Target Entities-######################
[Target Entities]:
- Alex
- Taylor
- Jordan
[Target Relationships]:
- Alex --&-- Taylor
- Alex --&-- Jordan
- Taylor --&-- Jordan

######################-Example of Anticipated Output Format-######################
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.){record_delimiter}
(entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device.{tuple_delimiter}power dynamics, perspective shift){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.{tuple_delimiter}shared goals, rebellion){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.{tuple_delimiter}conflict resolution, mutual respect){record_delimiter}

#############################-Real Data-######################
[Target Entities]:
{target_entities}
[Target Relationships]:
{target_relationships}
[Info]: {info}
######################
Output:
"""

PROMPTS["planning"] = """You are an intelligent assistant responsible for resolving the [Main Query] by making appropriate operations as specified.
With respect to the [Main Query], you have consolidated some relevant information in your [Memory] that records 
- **Previous Subqueries**: `A series of subqueries you previously raised to support the resolving of the [Main Query].`
- **Memory Points** `The existing memory points describing what you have already known regarding the [Main Query].`

-Goal-
Your task is to determine whether current [Memory] has been sufficient to comprehensively deal with the [Main Query] and make responses according to the following procedures.

-Procedures-
1. Analyze the [Main Query] and [Memory], and judge if the existing information in [Memory] is sufficient.

2. Make appropriate responses following the logic branches below.
Case 1: If the existing information in [Memory] is sufficient, output <None> in [New Subquery].
Case 2: If not, you should construct a new subquery that aims at retrieving missing evidences or investigating under-explored aspects.
    Case 2.1: If you need to delve deeper into specific memory points to retrieve more supplementary details, generate a constrained-form subquery and explicitly indicate the indices of target memory points. Format each subquery as (<subquery>{tuple_delimiter}<memory_point_indices>)
    Case 2.2: If you think that there are still under-explored aspects that go beyond the coverage of current [Memory], generate one free-form subquery to investigate from certain perspectives that could possibly provide new exploration directions.

3. Output your judgement in [Judgement] in the form of corresponding case index (1, 2.1 or 2.2), and give proper subquery/subqueries (if any) using the appropriate format according to different cases as **Example of Anticipated Output Format** given below.

* Note that:
(1) When generating new subquery, you should consider what you have known and what subqueries you have previously raised. 
(2) Avoid generating overly repetitive new subquery that cannot bring new exploration.
(3) Only output the judgement and corresponding new subquery without any additional content.


######################-Examples Memory Points-######################
- Point (0)
Involved Objects: Apple's Product Line{object_delimiter}iPhones{object_delimiter}iPads{object_delimiter}Macs{object_delimiter}Apple Watches{object_delimiter}AirPods{object_delimiter}Apple TV+{object_delimiter}Apple Music{object_delimiter}Apple Arcade
Description: Apple's product line encompasses a wide range of devices and services, including iPhones, iPads, Macs, Apple Watches, AirPods, and various accessories and software. They also offer services like Apple TV+, Apple Music, and Apple Arcade.
- Point (1)
Involved Objects: Steve Jobs{object_delimiter}San Francisco, California{object_delimiter}Paul{object_delimiter}Clara Jobs{object_delimiter}
Description: Steve Jobs was born on February 24, 1955, in San Francisco, California, and was adopted by Paul and Clara Jobs. He grew up in Mountain View, California, in what would later become Silicon Valley.
- Point (2)
Involved Objects: Pricing Strategy{object_delimiter}Premium Pricing{object_delimiter}Freemium Model{object_delimiter}Tiered Pricing Structures
Description: Apple's pricing strategy has evolved from a focus on premium, high-end products with a "cult following" to a more diversified approach that includes both premium and more affordable options like freemium model. This shift has involved strategies like price skimming for new releases, tiered pricing structures with various models at different price points, and even some instances of underpricing to attract new users.

###########-Example of Anticipated Output Format for Case 1-###########
[Judgement]: 1
[New Subquery]: <None>

###########-Example of Anticipated Output Format for Case 2.1-###########
[Judgement]: 2.1
[New Subquery]: What are the price levels of Apple's high-end products and other affordable options like freemium model?{tuple_delimiter}0,2

###########-Example of Anticipated Output Format for Case 2.2-###########
[Judgement]: 2.2
[New Subquery]: How did Steve Jobs improve Apple's production efficiency and market share when he was Apple's CEO?

######################-Real Data-######################
[Main Query]: {query}
[Memory]:
{memory}
######################
Output:
"""

PROMPTS["planning_with_hint"] = """You are an intelligent assistant responsible for dealing with the [Main Query] by making appropriate operations as specified.
With respect to the [Main Query], you have consolidated some memory points in your [Memory] describing what you have already known regarding the [Main Query].
Each memory point can be seen as a specific aspect relevant to the [Main Query], providing necessary details or insights from its perspective. 

-Goal-
Your task is to analyze the [Main Query] and [Memory], then determine whether current [Memory] has been sufficient to comprehensively resolve the [Main Query].
If not sufficient, you need to indicate what you want to further investigate.

-Procedures-
Step 1.
Make appropriate judgement following the logic branches below. At this step, you can refer to the [Extended Info] as hints, which contains potentially useful information to deal with the [Main Query].

Case 1: If the [Memory] has been sufficient to completely resolve the [Main Query], output <None> in [Concerns].
Case 2: If the [Memory] is not sufficient, determine current situation should be attributed to which of the following subcases.
    Case 2.1: There are some specific memory points which you want to further investigate more details about. 
    Case 2.2: There are unexplored aspects that go beyond the scope of current [Memory] (i.e. not related to any of the existing memory points). 

Step 2.
Output as **Example of Anticipated Output Format**.
Specifically, give your judgement in [Judgement] using corresponding case index (1, 2.1 or 2.2).
Then, generate several concerns that aim at exploring details or aspects not addressed by current [Memory] to better resolve the [Main Query]
    When case 2.1, generate up to {num_concerns} concerns, each of which targets at a specific memory point. For each concern, specify the index of its corresponding memory point.
    When case 2.2, generate up to {num_concerns} concerns that probe meaningful information not yet covered by current [Memory]

######################-Examples Memory Points-######################
- Point (0)
Involved Objects: Apple's Product Line{object_delimiter}iPhones{object_delimiter}iPads{object_delimiter}Macs{object_delimiter}Apple Watches{object_delimiter}AirPods{object_delimiter}Apple TV+{object_delimiter}Apple Music{object_delimiter}Apple Arcade
Description: Apple's product line encompasses a wide range of devices and services, including iPhones, iPads, Macs, Apple Watches, AirPods, and various accessories and software. They also offer services like Apple TV+, Apple Music, and Apple Arcade.
- Point (1)
Involved Objects: Steve Jobs{object_delimiter}San Francisco, California{object_delimiter}Paul{object_delimiter}Clara Jobs{object_delimiter}
Description: Steve Jobs was born on February 24, 1955, in San Francisco, California, and was adopted by Paul and Clara Jobs. He grew up in Mountain View, California, in what would later become Silicon Valley.
- Point (2)
Involved Objects: Pricing Strategy{object_delimiter}Premium Pricing{object_delimiter}Freemium Model{object_delimiter}Tiered Pricing Structures
Description: Apple's pricing strategy has evolved from a focus on premium, high-end products with a "cult following" to a more diversified approach that includes both premium and more affordable options like freemium model. This shift has involved strategies like price skimming for new releases, tiered pricing structures with various models at different price points, and even some instances of underpricing to attract new users.

###########-Example of Anticipated Output Format for Case 1-###########
[Judgement]: 1
[Concerns]: <None>

###########-Example of Anticipated Output Format for Case 2.1-###########
[Judgement]: 2.1
[Concerns]:
0{tuple_delimiter}your_concern_1{record_delimiter}
2{tuple_delimiter}your_concern_2{record_delimiter}
2{tuple_delimiter}your_concern_3{record_delimiter}
{completion_delimiter}

###########-Example of Anticipated Output Format for Case 2.2-###########
[Judgement]: 2.2
[Concerns]:
your_concern_1{record_delimiter}    
your_concern_2{record_delimiter}
your_concern_3{record_delimiter}
{completion_delimiter}

######################-Real Data-######################
[Main Query]: {query}

[Memory]:
{memory}

[Extended Info]:
{extended_info}

######################
* Note that:
(1) Your judgement should not be totally restricted by the [Extended Info] because there might be other useful entities not yet covered.
(2) Your concern should be concise and suggest what further details or aspect you subsequently will seek for.
(3) Only output the judgement, concerns, and the indices of corresponding memory points without any other content.
(4) If current [Memory] has covered most relevant perspectives, generate fewer concerns to avoid redundancy.
(5) Your generated concerns should be separated by "{record_delimiter}".

######################
Output:
"""

PROMPTS["rag_response_with_memory_system_prompt"] = """---Role---
You are a helpful assistant responding to the user's [Main Query] using the information in your memory and the provided data tables.

---Goal---
Generate a response that deals with the [Main Query] using all information in your ---Memory-- and the input ---Data tables---, and incorporating any relevant general knowledge.

---Memory---
{memory}

---Data tables---
{context_data}
"""

PROMPTS["rag_response_with_memory_user_prompt"] = """The user's [Main Query] is as below.
[Main Query]: {query}

If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
Your response should conform to the below ---Response Requirements---.
---Response Requirements---
{response_type}

Your response:
"""

PROMPTS["generate_subquery"] = """You are an assistant responsible for dealing with the [Main Query].
Although you have had some relevant information in your [Memory], your current [Memory] is still not sufficient to comprehensively resolve the [Main Query] due to the concern given in [Concern].
Therefore, you need to generate a subquery that aims at either retrieving more evidences or investigating unexplored aspects in [Subquery] to better deal with the [Main Query] ultimately.

Meanwhile, [Extended Info] contains more information related to your current [Memory], which may be used as useful hints for your subquery generation.

[Previous Subqueries] records a series of previous subqueries that have been raised before.

###########-Anticipated Output Format-###########
[Subquery]: xxx

######################-Real Data-######################
[Main Query]: {query}

[Memory]:
{memory}

[Concern]:
{concern}

[Previous Subqueries]:
{history_subqueries}

[Extended Info]:
{extended_info}
######################
* Note that:
(1) Your generated subquery should be concise and address the concerns in your [Concern].
(2) The scope of your generated subquery should not be totally restricted by the [Extended Info] because there might be other possibly useful entities not yet covered.
(3) You should avoid generating a subquery that is overly similar to any one of the [Previous Subqueries] or [Main Query].
(4) Only output your subquery without any other redundant content such as markup strings.
######################
Output:
"""

PROMPTS["generate_subqueries"] = """You are an assistant responsible for dealing with the [Main Query].
Although you have had some relevant information in your [Memory], your current [Memory] is still not sufficient to comprehensively resolve the [Main Query] due to the [Concerns].
For each concern in [Concerns], it either relates to a specific memory point or transcends the scope of current [Memory]. 

-Goal-
Your task is to generate subqueries that aim at either retrieving more evidences or investigating unexplored aspects elaborated in [Concerns] so as to ultimately resolve the [Main Query].

Critical Requirements:
(1) Semantic Differentiation: Ensure generated subqueries are semantically distinct from any one of [Previous Subqueries] or [Main Query].
(2) Complementary Coverage: All subqueries should aim at different information dimensions not addressed by [Previous Subqueries].
(3) Relevance Maintenance: All subqueries must be relevant to answering the [Main Query].
(4) Every concern in [Concerns] should have a corresponding subquery.
(5) Avoid generating any markup strings.

Meanwhile, [Extended Info] contains more information related to your current [Memory], which may be used as useful hints for your subquery generation.
[Previous Subqueries] records a series of previous subqueries that have been raised before.

###########-Anticipated Output Format-###########
[Subqueries]:
subquery_1{record_delimiter}
subquery_2{record_delimiter}
subquery_3{record_delimiter}
{completion_delimiter}

######################-Real Data-######################
[Main Query]: {query}

[Memory]:
{memory}

[Concerns]:
{concerns}

[Previous Subqueries]:
{history_subqueries}

[Extended Info]:
{extended_info}
######################
* Note that:
• Your generated subqueries should be concise and address corresponding concerns in your [Concerns] in the same order.
• The scope of your generated subqueries should not be totally restricted by the [Extended Info] because there might be other useful entities not yet covered.
• Your generated subqueries should be separated by "{record_delimiter}" and in one-to-one correspondence with the [Concerns].

######################
Output:
"""

PROMPTS["reorganize_memory"] = """For resolving the [Main Query], you have consolidated some memory points in your [Memory] recording the relevant information you have known.
Based on current [Memory], your task is to conduct memory reorganization that merges multiple memory points into new ones when they are more suitable to consitute a semantically/logically cohesive unit as a whole. 

Specifically, you need to specify the indices of original memory points to merge.
Then, for each newly merged point, provide updated descriptions that could build essentially higher-order associations while preserving their original information necessary for dealing with the [Main Query].

Format each reorganized memory point as <indices>{tuple_delimiter}<new_description>
Output in [Points_to_Merge] using {language} as the **Example of Anticipated Output Format**.

######################-Example of Anticipated Output Format-######################
[Points_to_Merge]:
(1,2{tuple_delimiter}<new_description>){record_delimiter}
(2,4,5{tuple_delimiter}<new_description>){record_delimiter}
{completion_delimiter}

######################-Real Data-######################
[Main Query]: {main_query}
[Memory]:
{memory}

######################
Note that, after reorganization,
(1) Each new memory point should encapsulate a semantically/logically cohesive unit that highlights unique and essential association among the involved entities of original memory points.
(2) Each memory point aims to cover distinct aspects, minimizing overlap across different memory points.
(3) Memory redundancy is reduced by eliminating duplicate content across different memory points.
(4) If an original point itself is more suitable to be kept as a separate point, just leave it unchanged and do not output it.
(5) If a memory point has included objects significantly more than other points and encapsulated comprehensive content, it should not be further merged.
(6) Avoid forcibly merging. If there is no suitable points to merge, output <None> in [Points_to_Merge] without any other thing.
(7) Principally, new memory points should primarily reuse original entities and preserve original useful details as much as possible.

Output:
"""

PROMPTS["select_entities"] = """
You will be provided with a [Query], a set of retrieved [Entity Candidates] and your current [Memory].
[Memory] contains the relevant information you have known.

Your task is to select those entities that are relevant and potentially useful to deal with the [Query].

Output comma-separated indices of your selected entities in ascending order in [Selected]. If no candidate is useful, just output <None>

Conform to the below -Example of Anticipated Output Format-.
######################-Example of Anticipated Output Format-######################
[Selected]: 1,2,5

######################-Real Case-######################"
[Query]: {query}

[Memory]:
{memory}

[Entity Candidates]:
{entities_str}
######################
Note that:
(1) If the detail of an entity has been totally included in current [Memory], it should not be selected again.
(2) Only output the indices of selected entities without explanation.
######################
Output:
"""