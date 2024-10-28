import speech_recognition as sr
import time
from gtts import gTTS  # new import
from io import BytesIO 
def speak(text):
    audio_bytes = BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()



def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio, language='en-in')

    except Exception as e:

        return "None"
    return query


class_9_subjects = {
    "Science": [
        "Chapter 1: Matter in Our Surroundings",
        "Chapter 2: Is Matter Around Us Pure",
        "Chapter 3: Atoms and Molecules",
        "Chapter 4: Structure of the Atom",
        "Chapter 5: The Fundamental Unit of Life",
        "Chapter 6: Tissues",
        "Chapter 7: Diversity in Living Organisms",
        "Chapter 8: Motion",
        "Chapter 9: Force and Laws of Motion",
        "Chapter 10: Gravitation",
        "Chapter 11: Work and Energy",
        "Chapter 12: Sound",
        "Chapter 13: Why Do We Fall Ill",
        "Chapter 14: Natural Resources",
        "Chapter 15: Improvement in Food Resources"
    ],
    "English": [
        "Beehive - Chapter 1: The Fun They Had",
        "Beehive - Chapter 2: The Sound of Music",
        "Beehive - Chapter 3: The Little Girl",
        "Beehive - Chapter 4: A Truly Beautiful Mind",
        "Beehive - Chapter 5: The Snake and the Mirror",
        "Beehive - Chapter 6: My Childhood",
        "Beehive - Chapter 7: Packing",
        "Beehive - Chapter 8: Reach for the Top",
        "Beehive - Chapter 9: The Bond of Love",
        "Beehive - Chapter 10: Kathmandu",
        "Beehive - Chapter 11: If I Were You",
        "Moments - Chapter 1: The Lost Child",
        "Moments - Chapter 2: The Adventure of Toto",
        "Moments - Chapter 3: Iswaran the Storyteller",
        "Moments - Chapter 4: In the Kingdom of Fools",
        "Moments - Chapter 5: The Happy Prince",
        "Moments - Chapter 6: Weathering the Storm in Ersama",
        "Moments - Chapter 7: The Last Leaf",
        "Moments - Chapter 8: A House is Not a Home",
        "Moments - Chapter 9: The Accidental Tourist",
        "Moments - Chapter 10: The Beggar"
    ],
    "History": [
        "Chapter 1: The French Revolution",
        "Chapter 2: Socialism in Europe and the Russian Revolution",
        "Chapter 3: Nazism and the Rise of Hitler",
        "Chapter 4: Forest Society and Colonialism",
        "Chapter 5: Pastoralists in the Modern World",
        "Chapter 6: Peasants and Farmers",
        "Chapter 7: History and Sport: The Story of Cricket",
        "Chapter 8: Clothing: A Social History"
    ],
    "Geography": [
        "Chapter 1: India - Size and Location",
        "Chapter 2: Physical Features of India",
        "Chapter 3: Drainage",
        "Chapter 4: Climate",
        "Chapter 5: Natural Vegetation and Wildlife",
        "Chapter 6: Population"
    ],
    "Civics": [
        "Chapter 1: What is Democracy? Why Democracy?",
        "Chapter 2: Constitutional Design",
        "Chapter 3: Electoral Politics",
        "Chapter 4: Working of Institutions",
        "Chapter 5: Democratic Rights"
    ],
    "Economics": [
        "Chapter 1: The Story of Village Palampur",
        "Chapter 2: People as Resource",
        "Chapter 3: Poverty as a Challenge",
        "Chapter 4: Food Security in India"
    ]
}

class_10_subjects = {
    "Science": [
        "Chapter 1: Chemical Reactions and Equations",
        "Chapter 2: Acids, Bases, and Salts",
        "Chapter 3: Metals and Non-Metals",
        "Chapter 4: Carbon and Its Compounds",
        "Chapter 5: Periodic Classification of Elements",
        "Chapter 6: Light",
        "Chapter 7: Human Eye and the Colourful World",
        "Chapter 8: Electricity",
        "Chapter 9: Magnetic Effects of Current",
        "Chapter 10: Sources of Energy",
        "Chapter 11: Life Processes",
        "Chapter 12: Control and Coordination",
        "Chapter 13: How do Organisms Reproduce?",
        "Chapter 14: Heredity and Evolution",
        "Chapter 15: Our Environment",
        "Chapter 16: Management of Natural Resources"
    ],
    "English": [
        "First Flight - Chapter 1: A Letter to God",
        "First Flight - Chapter 2: Nelson Mandela: Long Walk to Freedom",
        "First Flight - Chapter 3: Two Stories about Flying",
        "First Flight - Chapter 4: From the Diary of Anne Frank",
        "First Flight - Chapter 5: The Hundred Dresses - I",
        "First Flight - Chapter 6: The Hundred Dresses - II",
        "First Flight - Chapter 7: Glimpses of India",
        "First Flight - Chapter 8: Mijbil the Otter",
        "First Flight - Chapter 9: Madam Rides the Bus",
        "First Flight - Chapter 10: The Sermon at Benares",
        "First Flight - Chapter 11: The Proposal",
        "Footprints Without Feet - Chapter 1: A Triumph of Surgery",
        "Footprints Without Feet - Chapter 2: The Thief’s Story",
        "Footprints Without Feet - Chapter 3: The Midnight Visitor",
        "Footprints Without Feet - Chapter 4: A Question of Trust",
        "Footprints Without Feet - Chapter 5: The Book That Saved the Earth",
        "Footprints Without Feet - Chapter 6: The Drop of Blood",
        "Footprints Without Feet - Chapter 7: The Making of a Scientist",
        "Footprints Without Feet - Chapter 8: The Beggar"
    ],
    "History": [
        "Chapter 1: The Rise of Nationalism in Europe",
        "Chapter 2: Nationalism in India",
        "Chapter 3: The Making of a Global World",
        "Chapter 4: The Age of Industrialization",
        "Chapter 5: Print Culture and the Modern World",
        "Chapter 6: Novels, Society, and History"
    ],
    "Geography": [
        "Chapter 1: Resources and Development",
        "Chapter 2: Forest and Wildlife Resources",
        "Chapter 3: Water Resources",
        "Chapter 4: Agriculture",
        "Chapter 5: Minerals and Energy Resources",
        "Chapter 6: Manufacturing Industries",
        "Chapter 7: Life Lines of National Economy"
    ],
    "Civics": [
        "Chapter 1: Power Sharing",
        "Chapter 2: Federalism",
        "Chapter 3: Political Parties",
        "Chapter 4: Democratic Rights"
    ],
    "Economics": [
        "Chapter 1: Development",
        "Chapter 2: Sectors of the Indian Economy",
        "Chapter 3: Money and Credit",
        "Chapter 4: Globalization and the Indian Economy",
        "Chapter 5: Consumer Rights"
    ]
}

class_11_subjects = {
    "Biology": [
        "Chapter 1: Diversity in Living World",
        "Chapter 2: Structural Organisation in Animals and Plants",
        "Chapter 3: Cell Structure and Function",
        "Chapter 4: Plant Physiology",
        "Chapter 5: Human Physiology",
        "Chapter 6: Reproduction",
        "Chapter 7: Genetics and Evolution",
        "Chapter 8: Biology and Human Welfare",
        "Chapter 9: Biotechnology and its Applications",
        "Chapter 10: Ecology and Environment"
    ],
    "English": [
        "Hornbill - Chapter 1: The Portrait of a Lady",
        "Hornbill - Chapter 2: We're Not Afraid to Die... if We Can All Be Together",
        "Hornbill - Chapter 3: Discovering Tut: The Saga Continues",
        "Hornbill - Chapter 4: Landscape of the Soul",
        "Hornbill - Chapter 5: The Ailing Planet: The Green Movement's Role",
        "Hornbill - Chapter 6: The Browning Version",
        "Hornbill - Chapter 7: The Adventure",
        "Hornbill - Chapter 8: Silk Road",
        "Snapshots - Chapter 1: The Summer of the Beautiful White Horse",
        "Snapshots - Chapter 2: The Address",
        "Snapshots - Chapter 3: Ranga's Marriage",
        "Snapshots - Chapter 4: Albert Einstein at School",
        "Snapshots - Chapter 5: Mother’s Day",
        "Snapshots - Chapter 6: The Ghat of the Only World",
        "Snapshots - Chapter 7: A House is Not a Home",
        "Snapshots - Chapter 8: The Book That Saved the Earth"
    ],
    "Physics": [
        "Chapter 1: Physical World",
        "Chapter 2: Units and Measurements",
        "Chapter 3: Motion in a Straight Line",
        "Chapter 4: Motion in a Plane",
        "Chapter 5: Laws of Motion",
        "Chapter 6: Work, Energy, and Power",
        "Chapter 7: System of Particles and Rotational Motion",
        "Chapter 8: Gravitation",
        "Chapter 9: Properties of Bulk Matter",
        "Chapter 10: Thermodynamics",
        "Chapter 11: Behaviour of Perfect Gas and Kinetic Theory",
        "Chapter 12: Oscillations and Waves"
    ],
    "Chemistry": [
        "Chapter 1: Some Basic Concepts of Chemistry",
        "Chapter 2: Structure of Atom",
        "Chapter 3: Classification of Elements and Periodicity in Properties",
        "Chapter 4: Chemical Bonding and Molecular Structure",
        "Chapter 5: States of Matter: Gases and Liquids",
        "Chapter 6: Thermodynamics",
        "Chapter 7: Equilibrium",
        "Chapter 8: Redox Reactions",
        "Chapter 9: Hydrogen",
        "Chapter 10: s-Block Element (Alkali and Alkaline earth metals)",
        "Chapter 11: Some p-Block Elements",
        "Chapter 12: Organic Chemistry - Some Basic Principles and Techniques",
        "Chapter 13: Hydrocarbons",
        "Chapter 14: Environmental Chemistry"
    ]
}

class_12_subjects = {
    "Biology": [
        "Chapter 1: Reproduction",
        "Chapter 2: Genetics and Evolution",
        "Chapter 3: Biology and Human Welfare",
        "Chapter 4: Biotechnology and Its Applications",
        "Chapter 5: Ecology and Environment",
        "Chapter 6: Biotechnology: Principles and Processes",
        "Chapter 7: Human Health and Disease",
        "Chapter 8: Strategies for Enhancement in Food Production",
        "Chapter 9: Microbes in Human Welfare",
        "Chapter 10: Biodiversity and Conservation",
        "Chapter 11: Biotechnology and Its Applications",
        "Chapter 12: Organisms and Populations",
        "Chapter 13: Ecosystem",
        "Chapter 14: Environmental Issues"
    ],
    "English": [
        "Flamingo - Chapter 1: The Last Lesson",
        "Flamingo - Chapter 2: Lost Spring: Stories of Stolen Childhood",
        "Flamingo - Chapter 3: Deep Water",
        "Flamingo - Chapter 4: The Rattrap",
        "Flamingo - Chapter 5: Indigo",
        "Flamingo - Chapter 6: Going Places",
        "Vistas - Chapter 1: The Third Level",
        "Vistas - Chapter 2: The Tiger King",
        "Vistas - Chapter 3: Journey to the End of the Earth",
        "Vistas - Chapter 4: The Enemy",
        "Vistas - Chapter 5: Should Wizard Hit Mommy?",
        "Vistas - Chapter 6: On the Face of It",
        "Vistas - Chapter 7: Evans Tries an O-Level",
        "Vistas - Chapter 8: Memories of Childhood"
    ],
    "Physics": [
        "Chapter 1: Electric Charges and Fields",
        "Chapter 2: Electrostatic Potential and Capacitance",
        "Chapter 3: Current Electricity",
        "Chapter 4: Moving Charges and Magnetism",
        "Chapter 5: Magnetism and Matter",
        "Chapter 6: Electromagnetic Induction",
        "Chapter 7: Alternating Currents",
        "Chapter 8: Electromagnetic Waves",
        "Chapter 9: Optics",
        "Chapter 10: Wave Optics",
        "Chapter 11: Dual Nature of Radiation and Matter",
        "Chapter 12: Atoms",
        "Chapter 13: Nuclei",
        "Chapter 14: Semiconductor Electronics",
        "Chapter 15: Communication Systems"
    ],
    "Chemistry": [
        "Chapter 1: The Solid State",
        "Chapter 2: Solutions",
        "Chapter 3: Electrochemistry",
        "Chapter 4: Chemical Kinetics",
        "Chapter 5: Surface Chemistry",
        "Chapter 6: General Principles and Processes of Isolation of Elements",
        "Chapter 7: p-Block Elements",
        "Chapter 8: d and f Block Elements",
        "Chapter 9: Coordination Compounds",
        "Chapter 10: Haloalkanes and Haloarenes",
        "Chapter 11: Alcohols, Phenols, and Ethers",
        "Chapter 12: Aldehydes, Ketones, and Carboxylic Acids",
        "Chapter 13: Organic Compounds Containing Nitrogen",
        "Chapter 14: Biomolecules",
        "Chapter 15: Polymers",
        "Chapter 16: Chemistry in Everyday Life"
    ]
}

