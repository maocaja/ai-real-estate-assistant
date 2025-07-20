import httpx # A modern, async-friendly HTTP client
from typing import List, Dict, Any, Optional

from chat_api.config.settings import settings
from data_service.models.project import Project, ProjectAmenitiesData # Reusamos modelos del data-service

class DataServiceClient:
    def __init__(self):
        self.base_url = settings.DATA_SERVICE_URL
        # Usamos httpx.AsyncClient para operaciones asíncronas
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def get_filtered_projects(
        self,
        city: Optional[str] = None,
        property_type: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_bedrooms: Optional[int] = None,
        min_bathrooms: Optional[int] = None,
        min_area_sqm: Optional[float] = None,
        recommended_use: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> List[Project]:
        """Fetches projects from the data-service with applied filters."""
        params = {k: v for k, v in locals().items() if k not in ['self', 'client'] and v is not None}
        try:
            response = await self.client.get("/projects", params=params, timeout=30)
            response.raise_for_status()
            # Validar la respuesta con el modelo Project del data-service
            return [Project(**item) for item in response.json()]
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error fetching filtered projects from data-service: {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            print(f"❌ Network error fetching filtered projects from data-service: {e}")
            return []
        except Exception as e:
            print(f"❌ An unexpected error occurred while fetching filtered projects: {e}")
            return []

    async def get_project_by_id(self, project_id: str) -> Optional[Project]:
        """Fetches a single project's details by ID from the data-service."""
        try:
            response = await self.client.get(f"/projects/{project_id}", timeout=10)
            response.raise_for_status()
            print(f"Data from data-service for validation: {response.json().keys() if response.json() else 'None'}")
            return Project(**response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"Project with ID '{project_id}' not found in data-service.")
                return None
            print(f"❌ HTTP error fetching project by ID from data-service: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"❌ Network error fetching project by ID from data-service: {e}")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred while fetching project by ID: {e}")
            return None

    async def get_all_project_ids(self) -> List[str]:
        """Fetches all project IDs from the data-service."""
        try:
            response = await self.client.get("/projects/all_ids", timeout=10)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"❌ Network error fetching all project IDs from data-service: {e}")
            return []
        except Exception as e:
            print(f"❌ An unexpected error occurred while fetching all project IDs: {e}")
            return []

    async def get_financial_assessment(
        self,
        project_price: float,
        monthly_income: float,
        loan_term_years: int,
        interest_rate_annual: float = 12.0 # Tasa de interés predeterminada (12% anual)
    ) -> Dict[str, Any]:
        """
        Calcula una estimación de la cuota mensual de un préstamo hipotecario y el valor total a pagar.

        Args:
            project_price (float): Precio total del inmueble.
            monthly_income (float): Ingresos mensuales del solicitante.
            loan_term_years (int): Plazo del préstamo en años.
            interest_rate_annual (float): Tasa de interés anual en porcentaje (ej. 12 para 12%).

        Returns:
            Dict[str, Any]: Un diccionario con la cuota mensual estimada,
                            el valor total del préstamo, y el total pagado con intereses.
                            Retorna None o un diccionario vacío si los inputs no son válidos.
        """
        if project_price <= 0 or monthly_income <= 0 or loan_term_years <= 0 or interest_rate_annual <= 0:
            logger.warning("Invalid input for financial assessment.")
            return {"error": "Los valores de entrada deben ser positivos."}

        try:
            # Calcular la tasa de interés mensual
            monthly_interest_rate = (interest_rate_annual / 100) / 12

            # Número total de pagos
            num_payments = loan_term_years * 12

            # Asumir que el monto del préstamo es el 70% del valor del inmueble,
            # lo típico para un crédito hipotecario en Colombia.
            # Puedes ajustar esta lógica según las políticas que quieras simular.
            loan_amount = project_price * 0.70 # Simulación del monto del préstamo
            down_payment = project_price * 0.30 # Simulación de la cuota inicial

            # Cálculo de la cuota mensual usando la fórmula de anualidad de préstamo
            # M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]
            # Donde:
            # M = Pago mensual
            # P = Monto principal del préstamo (loan_amount)
            # i = Tasa de interés mensual
            # n = Número total de pagos
            if monthly_interest_rate == 0: # Evitar división por cero
                monthly_payment = loan_amount / num_payments
            else:
                monthly_payment = (
                    loan_amount
                    * (monthly_interest_rate * (1 + monthly_interest_rate) ** num_payments)
                    / ((1 + monthly_interest_rate) ** num_payments - 1)
                )

            total_paid_with_interest = monthly_payment * num_payments

            # Verificar asequibilidad (ejemplo: la cuota mensual no debe exceder el 30% del ingreso mensual)
            if monthly_payment > (monthly_income * 0.30):
                affordability_message = "La cuota mensual estimada excede el 30% de tus ingresos, lo que podría dificultar la aprobación del crédito. Considera un plazo más largo o un proyecto de menor valor."
            else:
                affordability_message = "La cuota mensual estimada parece asequible con tus ingresos."

            return {
                "project_price": project_price,
                "loan_amount": round(loan_amount, 2),
                "down_payment": round(down_payment, 2),
                "monthly_payment": round(monthly_payment, 2),
                "total_paid_with_interest": round(total_paid_with_interest, 2),
                "loan_term_years": loan_term_years,
                "annual_interest_rate": interest_rate_annual,
                "affordability_message": affordability_message
            }
        except Exception as e:
            logger.error(f"Error calculating financial assessment: {e}")
            return {"error": f"No pude realizar el cálculo financiero debido a un error: {e}"}
