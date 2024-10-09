from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import logging

# Configure logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')

class Metrics:
    """
    This class provides methods to evaluate machine learning models by calculating various performance metrics.
    
    Author: Sanket Jagtap
    """

    def __init__(self):
        """
        Initialize the Metrics class. Currently, no initialization is required.
        """
        pass

    def r2_score(self, y_true, y_pred):
        """
        Calculate the R-squared (R2) score of the model.

        The R2 score represents the proportion of variance in the dependent variable
        that is predictable from the independent variable(s). It ranges from 0 to 1,
        where 1 indicates perfect prediction.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: R2 score of the model.

        Raises:
            Exception: If calculation fails, logs the error and re-raises.
        """
        logging.info('Entered the r2_score method of the Metrics class')
        try:
            score = r2_score(y_true, y_pred)
            logging.info('Calculated r2_score. Exited the r2_score method of the Metrics class')
            return score
        except Exception as e:
            logging.error(f'Exception in r2_score method: {str(e)}')
            raise

    def adj_r2_score(self, x, y_true, y_pred):
        """
        Calculate the adjusted R-squared score of the model.

        The adjusted R2 score modifies the R2 score to account for the number of predictors
        in the model. It penalizes the addition of extraneous predictors.

        Args:
            x (array-like): Input features.
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: Adjusted R2 score of the model.

        Raises:
            Exception: If calculation fails, logs the error and re-raises.
        """
        logging.info('Entered the adj_r2_score method of the Metrics class')
        try:
            r2 = r2_score(y_true, y_pred)
            n = x.shape[0]  # Number of observations
            p = x.shape[1]  # Number of predictors
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            logging.info('Calculated adj_r2_score. Exited the adj_r2_score method of the Metrics class')
            return adj_r2
        except Exception as e:
            logging.error(f'Exception in adj_r2_score method: {str(e)}')
            raise

    def rmse_score(self, y_true, y_pred):
        """
        Calculate the Root Mean Square Error (RMSE) of the model.

        RMSE is the square root of the average of squared differences between 
        prediction and actual observation. It gives an idea of how much error 
        the system typically makes in its predictions, with a lower value indicating better fit.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: RMSE score of the model.

        Raises:
            Exception: If calculation fails, logs the error and re-raises.
        """
        logging.info('Entered the rmse_score method of the Metrics class')
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info('Calculated rmse_score. Exited the rmse_score method of the Metrics class')
            return rmse
        except Exception as e:
            logging.error(f'Exception in rmse_score method: {str(e)}')
            raise
