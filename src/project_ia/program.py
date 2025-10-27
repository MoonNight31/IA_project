import statistics
import matplotlib.pyplot as plt
import pandas

from pathlib import Path
from csv import DictReader, DictWriter
from random import randint, random, uniform, choice


from dataclasses import dataclass

@dataclass
class Car:
    manufacturer: str
    model: str
    year: int
    power: int
    torque: int
    max_speed: int
    fuel_efficiency: float  # L/100km équivalent
    fuel_type: str
    doors_number: int
    weight: int
    aerodynamic_level: float
    turbo_count: int
    millage_in_km: int
    zero_to_hundred: float
    transmission_type: str
    is_started: bool = False

    def start(self):
        self.is_started = True

    def stop(self):
        self.is_started = False


def generate_car() -> Car:

    models_per_brands = {
        "Renault": ["Clio", "Twingo", "Megane"],
        "Citroen": ["C3", "C4", "C5"],
        "Peugeot": ["208", "308", "3008"],
        "Mercedes": ["A-Class", "C-Class", "E-Class"],
        "Audi": ["A1", "A3", "A6"],
        "BMW": ["Serie 1", "Serie 3", "Serie 5"],
        "Ferrari": ["F8", "SF90", "Roma"],
        "Bugatti": ["Chiron", "Veyron", "Divo"],
        "Koenigsegg": ["Agera", "Regera", "Jesko"]
    }

    manufacturer = choice(list(models_per_brands.keys()))
    model = choice(models_per_brands[manufacturer])

    # Génération de l'année avec une distribution plus réaliste (plus de voitures récentes)
    year = randint(1995, 2025)

    doors_number = choice([2, 3, 4, 5])

    # --- Catégorisation par constructeur ---
    if manufacturer in ["Renault", "Citroen", "Peugeot"]:
        category = "citadine"
        engine_size = round(uniform(0.8, 1.6), 1)  # Litres
        doors_number = choice([3, 4, 5])
        power = randint(60, 120)
        weight = randint(900, 1200)
        aerodynamic_level = uniform(0.2, 0.35)
        # Plus de chances d'avoir de l'essence pour les citadines
        fuel_type = choice(["Gasoline"] * 7 + ["Diesel"] * 2 + ["Electric"])
        transmission_type = choice(["Manual"] * 5 + ["Automatic"])
    elif manufacturer in ["Mercedes", "Audi", "BMW"]:
        category = "berline"
        engine_size = round(uniform(1.5, 3.0), 1)
        doors_number = choice([3, 4, 5])
        power = randint(150, 350)
        weight = randint(1400, 2000)
        aerodynamic_level = uniform(0.3, 0.45)
        fuel_type = choice(["Gasoline"] * 5 + ["Diesel"] * 3 + ["Electric"] * 2)
        transmission_type = choice(["Manual"] * 3 + ["Automatic"] * 2)
    else:
        doors_number = choice([2, 3])
        category = "sportive"
        engine_size = round(uniform(3.0, 6.5), 1)
        power = randint(400, 1200)
        weight = randint(1100, 1800)
        aerodynamic_level = uniform(0.3, 0.4)
        fuel_type = choice(["Gasoline"] * 8 + ["Electric"] * 2)
        transmission_type = choice(["Manual"] + ["Automatic"] * 3)

    # --- Calculs dérivés cohérents ---
    # Torque plus réaliste basé sur la taille du moteur et la puissance
    torque = int(power * (engine_size / 2.5) * uniform(0.8, 1.1))


    # --- Turbo logic ---
    if fuel_type == "Electric":
        turbo_count = 0
    elif fuel_type == "Diesel":
        # Tous les diesels modernes ont un turbo
        if year < 2000:
            turbo_count = 1
        else:
            turbo_count = choice([1, 2])  # majoritairement 1
    else:  # essence
        if power < 120:
            turbo_count = 0 if year < 2010 else 1
        elif power < 300:
            turbo_count = choice([1, 1, 2])
        else:
            turbo_count = randint(2, 4) if manufacturer in ["Ferrari", "Bugatti", "Koenigsegg"] else choice([1, 2])

    # --- Millage avec une distribution plus réaliste ---
    age = 2024 - year
    if age > 0:
        # Simulation d'une distribution normale centrée sur 15000 km par an
        average_km_per_year = 15_000
        std_dev = 7_000  # Déviation standard
        millage_in_km = int(max(0, (age * average_km_per_year) + (random() - 0.5) * 2 * std_dev))
    else:
        millage_in_km = 0

    max_speed = int(min(120 + power * (0.55 - aerodynamic_level / 2), 500))


    # Fuel efficiency : litres/100km ou équivalent
    if fuel_type == "Gasoline":
        fuel_efficiency = round(uniform(5.0, 9.0) * (weight / 1500), 1)
    elif fuel_type == "Diesel":
        fuel_efficiency = round(uniform(4.0, 6.0) * (weight / 1500), 1)
    else:
        fuel_efficiency = round(uniform(15.0, 25.0) * (weight / 1500), 1)  # kWh/100km équivalent

    # Calcul basé sur le ratio puissance/poids
    power_to_weight = power / weight  # ex: 0.08 = citadine, 0.5 = supercar
    zero_to_hundred = round(2.5 / (power_to_weight ** 0.5), 1)

    # Retourner l'objet Car
    return Car(
        manufacturer, model, year, power, torque, max_speed,
        fuel_efficiency, fuel_type, doors_number, weight,
        aerodynamic_level, turbo_count, millage_in_km, zero_to_hundred,
        transmission_type
    )

class CarsLoader:
    @staticmethod
    def save(car: Car, csv_file: Path):
        contains_lines: bool = False
        with open(csv_file, "r") as ro_file:
            contains_lines = len(ro_file.readlines()) > 0
        with open(csv_file, "a") as file:
            writer: DictWriter = DictWriter(file, fieldnames=car.__dict__.keys())
            if (not contains_lines):
                writer.writeheader()
            writer.writerow(car.__dict__)

    @staticmethod
    def save_batch(cars: list[Car], csv_file: Path):
        """Sauvegarde plusieurs voitures en une seule fois."""
        file_exists_and_not_empty = csv_file.exists() and csv_file.stat().st_size > 0
        
        with open(csv_file, "a", newline='') as file:
            if cars:
                writer = DictWriter(file, fieldnames=cars[0].__dict__.keys())
                if not file_exists_and_not_empty:
                    writer.writeheader()
                for car in cars:
                    writer.writerow(car.__dict__)
    
    @staticmethod
    def load(csv_file: Path) -> list[Car]:
        with open(csv_file, "r") as file:
            reader: DictReader = DictReader(file)
            cars: list[Car] = []
            for row in reader:
                # Conversion des types car CSV lit tout en string
                try:
                    converted_row = {
                        'manufacturer': row['manufacturer'],
                        'model': row['model'],
                        'year': int(row['year']),
                        'power': int(row['power']),
                        'torque': int(row['torque']),
                        'max_speed': int(row['max_speed']),
                        'fuel_efficiency': float(row['fuel_efficiency']),
                        'fuel_type': row['fuel_type'],
                        'doors_number': int(row['doors_number']),
                        'weight': int(row['weight']),
                        'aerodynamic_level': float(row['aerodynamic_level']),
                        'turbo_count': int(row['turbo_count']),
                        'millage_in_km': int(row['millage_in_km']),
                        'zero_to_hundred': float(row['zero_to_hundred']),
                        'transmission_type': row['transmission_type'],
                        'is_started': row['is_started'].lower() == 'true'
                    }
                    car: Car = Car(**converted_row)
                    cars.append(car)
                except (ValueError, KeyError) as e:
                    print(f"Erreur lors du chargement d'une ligne: {e}")
                    continue
            return cars


def main():
    csv_file = Path("cars.csv")
    garage: list[Car] = []
    
    # Générer toutes les voitures
    for _ in range(149_000):
        car: Car = generate_car()
        garage.append(car)
    
    # Sauvegarder en une seule fois
    CarsLoader.save_batch(garage, csv_file)

    # Cette ligne était commentée avant - c'est pourquoi vous n'aviez pas l'erreur
    #garage_as_dict = [car.__dict__ for car in garage   ]
    #data_frame = pandas.DataFrame(garage_as_dict)
    #print(data_frame)
    
    # Si vous décommentez ces lignes, vous verrez l'erreur :
    # loaded_cars = CarsLoader.load(csv_file)
    # print(f"Voitures chargées: {len(loaded_cars)}")

if (__name__ == "__main__"):
    main()