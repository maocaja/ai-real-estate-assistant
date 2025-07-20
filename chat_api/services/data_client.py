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
        interest_rate_annual: float = 12.0, # Tasa de interés predeterminada (12% anual)
        expected_rental_income_monthly: Optional[float] = None, # Nuevo parámetro
        expected_annual_appreciation_rate: Optional[float] = None, # Nuevo parámetro
        annual_operating_costs_percentage: Optional[float] = 1.0, # Nuevo parámetro, con valor por defecto
        investment_horizon_years: Optional[int] = 10 # Nuevo parámetro, con valor por defecto
    ) -> Dict[str, Any]:
        """
        Calcula una estimación de la cuota mensual de un préstamo hipotecario y el valor total a pagar,
        además de un análisis básico de inversión (ROI y flujo de caja).

        Args:
            project_price (float): Precio total del inmueble.
            monthly_income (float): Ingresos mensuales del solicitante.
            loan_term_years (int): Plazo del préstamo en años.
            interest_rate_annual (float): Tasa de interés anual en porcentaje.
            expected_rental_income_monthly (Optional[float]): Ingreso de alquiler mensual estimado.
            expected_annual_appreciation_rate (Optional[float]): Tasa de apreciación anual esperada en porcentaje.
            annual_operating_costs_percentage (Optional[float]): Porcentaje anual del precio del proyecto para costos operativos.
            investment_horizon_years (Optional[int]): Horizonte de tiempo en años para el cálculo de inversión.

        Returns:
            Dict[str, Any]: Un diccionario con la cuota mensual estimada,
                            el valor total del préstamo, el total pagado con intereses,
                            y métricas de inversión como ROI y flujo de caja.
        """
        if project_price <= 0 or monthly_income <= 0 or loan_term_years <= 0 or interest_rate_annual <= 0:
            logger.warning("Invalid input for financial assessment.")
            return {"error": "Los valores de entrada principales (precio, ingresos, plazo, tasa de interés) deben ser positivos."}

        try:
            # --- Cálculos de Préstamo (Existentes) ---
            monthly_interest_rate = (interest_rate_annual / 100) / 12
            num_payments = loan_term_years * 12
            
            loan_amount_percentage = 0.70 # 70% del valor del inmueble
            loan_amount = project_price * loan_amount_percentage
            down_payment = project_price * (1 - loan_amount_percentage)

            if monthly_interest_rate == 0:
                monthly_loan_payment = loan_amount / num_payments
            else:
                monthly_loan_payment = (
                    loan_amount
                    * (monthly_interest_rate * (1 + monthly_interest_rate) ** num_payments)
                    / ((1 + monthly_interest_rate) ** num_payments - 1)
                )

            total_paid_with_interest = monthly_loan_payment * num_payments

            affordability_message = ""
            if monthly_loan_payment > (monthly_income * 0.30):
                affordability_message = "La cuota mensual estimada excede el 30% de tus ingresos, lo que podría dificultar la aprobación del crédito. Considera un plazo más largo o un proyecto de menor valor."
            else:
                affordability_message = "La cuota mensual estimada parece asequible con tus ingresos."

            result = {
                "project_price": project_price,
                "loan_amount": round(loan_amount, 2),
                "down_payment": round(down_payment, 2),
                "monthly_loan_payment": round(monthly_loan_payment, 2), # Renombrado para claridad
                "total_paid_with_interest": round(total_paid_with_interest, 2),
                "loan_term_years": loan_term_years,
                "annual_interest_rate": interest_rate_annual,
                "affordability_message": affordability_message
            }

            # --- Cálculos de Inversión (NUEVOS) ---
            if (expected_rental_income_monthly is not None and expected_rental_income_monthly > 0) and \
               (expected_annual_appreciation_rate is not None and expected_annual_appreciation_rate >= 0) and \
               (annual_operating_costs_percentage is not None and annual_operating_costs_percentage >= 0) and \
               (investment_horizon_years is not None and investment_horizon_years > 0):
                
                # Convertir porcentajes a decimales
                appreciation_rate_decimal = expected_annual_appreciation_rate / 100
                annual_operating_costs_decimal = annual_operating_costs_percentage / 100

                # Costos operativos anuales (basados en un porcentaje del precio del proyecto)
                annual_operating_costs = project_price * annual_operating_costs_decimal

                # Ingreso neto anual por alquiler (ingreso alquiler - cuota préstamo - costos operativos)
                # Asumimos que la cuota del préstamo es un gasto de la inversión para el flujo de caja
                net_rental_income_annual = (expected_rental_income_monthly * 12) - (monthly_loan_payment * 12) - annual_operating_costs
                
                # Valor futuro del inmueble (después del horizonte de inversión)
                future_property_value = project_price * ((1 + appreciation_rate_decimal) ** investment_horizon_years)

                # ROI (Return on Investment)
                # ROI = (Ganancia de la Inversión - Costo de la Inversión) / Costo de la Inversión
                # Ganancia de la inversión = (Valor futuro del inmueble - Monto del préstamo restante si se vende) + (Ingresos netos por alquiler acumulados)
                # Simplificación para este ROI básico: (Valor futuro - Inversión inicial total) / Inversión inicial total
                # Inversión inicial total = Cuota inicial + (posibles gastos de cierre, que aquí no consideramos)
                
                # ROI más simple: (Ganancia por apreciación + Ingresos netos acumulados) / Inversión inicial
                # Inversión inicial = Cuota inicial (down_payment)
                
                # Ganancia total = (Apreciación del valor) + (Ingresos netos por alquiler acumulados)
                # Apreciación del valor = future_property_value - project_price
                
                # Flujo de caja anual (simplificado)
                annual_cash_flow = (expected_rental_income_monthly * 12) - annual_operating_costs - (monthly_loan_payment * 12)

                # Calculo de ROI: (Ganancia total - Inversión inicial) / Inversión inicial
                # Donde Ganancia total = (future_property_value - project_price) + (net_rental_income_annual * investment_horizon_years)
                # Y Inversión inicial = down_payment (la cuota inicial)
                
                # Asegurarse de que la inversión inicial no sea cero para evitar división por cero
                if down_payment > 0:
                    total_gain = (future_property_value - project_price) + (net_rental_income_annual * investment_horizon_years)
                    roi_percentage = (total_gain / down_payment) * 100
                else:
                    roi_percentage = 0 # O manejar como un caso especial si no hay cuota inicial

                result.update({
                    "expected_rental_income_monthly": expected_rental_income_monthly,
                    "expected_annual_appreciation_rate": expected_annual_appreciation_rate,
                    "annual_operating_costs_percentage": annual_operating_costs_percentage,
                    "investment_horizon_years": investment_horizon_years,
                    "future_property_value": round(future_property_value, 2),
                    "annual_cash_flow": round(annual_cash_flow, 2),
                    "roi_percentage": round(roi_percentage, 2)
                })
            else:
                result["investment_analysis_message"] = "Se requiere el ingreso de alquiler mensual, la tasa de apreciación anual y el horizonte de inversión para realizar un análisis de ROI y flujo de caja."

            return result
        except Exception as e:
            logger.error(f"Error calculating financial assessment: {e}")
            return {"error": f"No pude realizar el cálculo financiero debido a un error: {e}"}

