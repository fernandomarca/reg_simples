#![allow(dead_code)]

use polars::prelude::*;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::model_selection::train_test_split;
use std::error::Error;
use std::fs::File;
use std::io::Write;

fn create_model() -> Result<(), Box<dyn Error>> {
    let data = load_data().expect("Erro ao carregar os dados");
    let model = train_model(data)?;
    save_model(&model)
}

fn load_data() -> Result<(Vec<f64>, Vec<i64>), Box<dyn Error>> {
    // Leia o arquivo CSV em um DataFrame
    let df = CsvReadOptions::default()
        .with_infer_schema_length(None)
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("pontuacao_teste.csv".into()))?
        .finish()?;
    println!("{}", df);

    let mx_vec: Vec<f64> = df
        .column("horas_estudo")
        .unwrap()
        .f64()
        .unwrap()
        .cont_slice()
        .unwrap()
        .to_vec();

    let my_vec: Vec<i64> = df
        .column("pontuacao_teste")
        .unwrap()
        .i64()
        .unwrap()
        .cont_slice()
        .unwrap()
        .to_vec();

    Ok((mx_vec, my_vec))
}

fn train_model(data: (Vec<f64>, Vec<i64>)) -> Result<LinearRegressionModel, Box<dyn Error>> {
    let (mx_vec, my_vec) = data;

    let mx = DenseMatrix::new(mx_vec.len(), 1, mx_vec, false);

    // 70% dos dados para treinamento e 30% para teste
    let (train_x, test_x, train_y, test_y) = train_test_split(&mx, &my_vec, 0.3, false, Some(50));

    // Create a Linear Regression model
    let model = LinearRegression::fit(&train_x, &train_y, Default::default())?;

    // y = ax + b
    let intercept = model.intercept();
    let coefficients = model.coefficients();
    let cof = coefficients.get((0, 0));
    println!("y = {}x + {}", cof, intercept);

    let y_pred = model.predict(&test_x)?;

    let test_y: Vec<f64> = test_y.into_iter().map(|x| x as f64).collect();
    let y_pred: Vec<f64> = y_pred.into_iter().map(|x| x as f64).collect();

    // r2 score
    let r2 = smartcore::metrics::r2(&test_y, &y_pred);
    println!("r2: {}", r2);

    // mean absolute error
    let mae = smartcore::metrics::mean_absolute_error(&test_y, &y_pred);
    println!("mae: {}", mae);

    // mean squared error
    let mse = smartcore::metrics::mean_squared_error(&test_y, &y_pred);
    println!("mse: {}", mse);

    // root mean squared error
    let rmse = mse.sqrt();
    println!("rmse: {}", rmse);

    Ok(model)
}

pub type LinearRegressionModel = LinearRegression<f64, i64, DenseMatrix<f64>, Vec<i64>>;

fn save_model(model: &LinearRegressionModel) -> Result<(), Box<dyn Error>> {
    let serialized_model = serde_json::to_string(&model)?;
    let mut file = File::create("model.json")?;
    file.write_all(serialized_model.as_bytes())?;
    Ok(())
}

pub fn load_model(path: &str) -> Result<LinearRegressionModel, Box<dyn Error>> {
    let file = File::open(path)?;
    let model: LinearRegressionModel = serde_json::from_reader(file).unwrap();
    Ok(model)
}

// let data = vec![vec![1.0], vec![3.0], vec![5.0], vec![7.0], vec![9.0]];

// let x = DenseMatrix::from_2d_vec(&data);
// let file = fs::File::open("pontuacao_teste.csv").unwrap();

// let mx: DenseMatrix<f64> = csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
//     file,
//     csv::CSVDefinition::default(),
// )
// .unwrap();
