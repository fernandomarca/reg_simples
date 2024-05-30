use crate::model::LinearRegressionModel;
use actix_web::{
    post,
    web::{self, Data},
    App, HttpResponse, HttpServer, Responder,
};
use serde::Deserialize;
use smartcore::linalg::basic::{arrays::Array, matrix::DenseMatrix};
use std::sync::Arc;

pub mod model;

#[derive(Deserialize, Debug)]
struct HoursRequestBody {
    hours: f64,
}

#[derive(Deserialize, Debug)]
struct PointsRequestBody {
    points: u64,
}

#[post("/predict")]
async fn predict(
    data: web::Json<HoursRequestBody>,
    app_state: Data<Arc<AppState>>,
) -> impl Responder {
    let input = DenseMatrix::new(1, 1, vec![data.hours], false);
    let y_pred = app_state.model.predict(&input).map_err(|e| {
        println!("Erro ao prever: {:?}", e);
        HttpResponse::InternalServerError().finish()
    });
    match y_pred {
        Ok(y_pred) => HttpResponse::Ok().json(y_pred[0]),
        Err(response) => response,
    }
}

#[post("/expected")]
async fn expected(
    data: web::Json<PointsRequestBody>,
    app_state: Data<Arc<AppState>>,
) -> impl Responder {
    let pontos = data.points;

    let intercept = app_state.model.intercept();
    let coefficients = app_state.model.coefficients();
    let cof = coefficients.get((0, 0));

    let horas = (pontos as f64 - intercept) / cof;

    HttpResponse::Ok().json(horas as u64)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = model::load_model("model.json").expect("Erro ao carregar o modelo");
    let app_state = Arc::new(AppState { model });

    HttpServer::new(move || {
        App::new()
            .app_data(Data::new(app_state.clone()))
            .service(predict)
            .service(expected)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

struct AppState {
    model: LinearRegressionModel,
}
