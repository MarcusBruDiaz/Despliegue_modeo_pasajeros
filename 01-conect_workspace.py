from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

interactive_auth = InteractiveLoginAuthentication(tenant_id="99e1e721-7184-498e-8aff-b2ad4e53c1c2")
ws = Workspace.get(name='mlw-esp-udeamb',
            subscription_id='d30f798a-582c-467d-a521-8768c9fd7ef4',
            resource_group='rg-ml-udea',
            location='eastus',
            auth=interactive_auth
            )

ws.write_config(path='.azureml')