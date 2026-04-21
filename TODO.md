# TODO.md - Fix Stable Predictions in Streamlit App

Approved plan to regenerate full dataset_modelamiento.csv with engineered features so predictions vary by input sliders.

## Steps:
- [ ] 1. Load retornos_diarios.csv
- [ ] 2. Run feature_engineering.calcular_volatilidad_historica (20d)
- [ ] 3. Run feature_engineering.calcular_momentum (5d)
- [ ] 4. Run feature_engineering.calcular_correlacion_rodante_brent (30d)
- [ ] 5. Run feature_engineering.calcular_delta_vix()
- [ ] 6. Run feature_engineering.crear_indicador_ventana()
- [ ] 7. Add 'sector' column
- [ ] 8. construir_dataset_modelamiento() → save CSV
- [ ] 9. Test app: vary sliders → confirm proba changes 40-80%
- [ ] 10. Update notebooks 02/03 if needed

Progress: 0/10 completed.

Next: Execute Python script for steps 1-8.

