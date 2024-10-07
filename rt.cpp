// rt: un lanzador de rayos minimalista
// g++ -O3 -fopenmp rt.cpp -o rt
#include <cmath> //nota: clang si es compatible con cmath (no es necesario)
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <chrono>
#include <sys/random.h>

//seed global
unsigned short seed[3] = {0,0,0};

class Vector 
{
public:        
	double x, y, z; // coordenadas x,y,z 
  
	// Constructor del vector, parametros por default en cero
	Vector(double x_= 0, double y_= 0, double z_= 0){ x=x_; y=y_; z=z_; }
  
	// operador para suma y resta de vectores
	Vector operator+(const Vector &b) const { return Vector(x + b.x, y + b.y, z + b.z); }
	Vector operator-(const Vector &b) const { return Vector(x - b.x, y - b.y, z - b.z); }
	// operator multiplicacion vector y escalar 
	Vector operator*(double b) const { return Vector(x * b, y * b, z * b); }
  
	// operator % para producto cruz
	Vector operator%(Vector&b){return Vector(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);}
	
	// producto punto con vector b
	double dot(const Vector &b) const { return x * b.x + y * b.y + z * b.z; }

	// producto elemento a elemento (Hadamard product)
	Vector mult(const Vector &b) const { return Vector(x * b.x, y * b.y, z * b.z); }
	
	// normalizar vector 
	Vector& normalize(){ return *this = *this * (1.0/sqrt(x * x + y * y + z * z)); }

};
typedef Vector Point;
typedef Vector Color;

class Ray 
{ 
public:
	Point o;
	Vector d; // origen y direcccion del rayo
	Ray(Point o_, Vector d_) : o(o_), d(d_) {} // constructor
};

class Sphere 
{
public:
	double r;	// radio de la esfera
	Point p;	// posicion
	Color c;	// color  
	Color radiance; //radiancia > 0 si es fuente de luz 
	int material; //material 0 lambertiano 1 microfacet
	Color eta;
	Color kappa;
	double alpha; //aspereza

	Sphere(double r_, Point p_, Color c_, Color radiance, int material, Color eta, Color kappa, double alpha): r(r_), p(p_), c(c_), radiance(radiance), material(material), eta(eta), kappa(kappa), alpha(alpha){}
  
	// PROYECTO 1
	// determina si el rayo intersecta a esta esfera
	double intersect(const Ray &ray) const { 
		// regresar distancia si hay intersección
		double t1,t2;
		double det=(ray.o-this->p).dot(ray.d)*(ray.o-this->p).dot(ray.d)-(ray.o-this->p).dot(ray.o-this->p)+this->r*this->r;
		if (det<0) return 0.0; // regresar 0.0 si no hay interseccion return 0.0;
		t2 = -(ray.o-this->p).dot(ray.d)+sqrt(det);
		t1 = -(ray.o-this->p).dot(ray.d)-sqrt(det);
		if(t1<0 || abs(t1)<0.0001) return t2; //caso especial donde la primera interseccion es invalida
		return t1; //t1 siempre es la primera interseccion con excepcion del caso especial 
		//los numeros negativos se pueden filtrar aqui o en bool intersect()
	}
	//version de intersect para vpt, regresa ambas intersecciones
	void intersectVPT(const Ray &ray, double &t1, double &t2) const { 
		// regresar distancia si hay intersección
		double det=(ray.o-this->p).dot(ray.d)*(ray.o-this->p).dot(ray.d)-(ray.o-this->p).dot(ray.o-this->p)+this->r*this->r;
		if (det<0) {t1=0.0; t2=0.0; return;} // regresar 0.0 si no hay interseccion return 0.0;
		t2 = -(ray.o-this->p).dot(ray.d)+sqrt(det);
		t1 = -(ray.o-this->p).dot(ray.d)-sqrt(det);
	}
};

Sphere spheres[] = {
	/*
	//Escena: radio, posicion, color, radiancia, material, eta, kappa, alpha
	Sphere(1e5,  Point(-1e5 - 49, 0, 0),  Color(.75, .25, .25),    Color (0, 0, 0),0, Color(), Color(),0), // left wall
  	Sphere(1e5,  Point(1e5 + 49, 0, 0),  Color(.25, .25, .75),    Color (0, 0, 0),0, Color(), Color(),0), // right wall
  	Sphere(1e5,  Point(0, 0, -1e5 - 81.6),  Color(.25, .75, .25),    Color (0, 0, 0),0, Color(), Color(),0), // back wall
  	Sphere(1e5,  Point(0, -1e5 - 40.8, 0),  Color(.25, .75, .75),    Color (0, 0, 0),0, Color(), Color(),0), // floor
  	Sphere(1e5,  Point(0, 1e5 + 40.8, 0),  Color(.75, .75, .25),     Color (0, 0, 0),0, Color(), Color(),0), // ceiling
  	Sphere(16.5,  Point(-23, -24.3, -34.6),  Color(.2, .3, .4),    Color (0, 0, 0),0, Color(), Color(),0), // bottom left sphere
  	Sphere(16.5,  Point(23, -24.3, -3.6),  Color(.4, .3, .2),    Color (0, 0, 0),0, Color(), Color(),0), // bottom right sphere
  	Sphere(5,  Point(14, -24.3, -35),  Color(),    Color (12, 12, 12),0, Color(), Color(),0) // light source
	
*/
/*
	Sphere(1e5,  Point(-1e5 - 49, 0, 0),  Color(.75, .25, .25),    Color (0, 0, 0),0, Color(), Color(),0), // left wall
  	Sphere(1e5,  Point(1e5 + 49, 0, 0),  Color(.25, .25, .75),    Color (0, 0, 0),0, Color(), Color(),0), // right wall
  	Sphere(1e5,  Point(0, 0, -1e5 - 81.6),  Color(.25, .75, .25),    Color (0, 0, 0),0, Color(), Color(),0), // back wall
  	Sphere(1e5,  Point(0, -1e5 - 40.8, 0),  Color(.25, .75, .75),    Color (0, 0, 0),0, Color(), Color(),0), // floor
  	Sphere(1e5,  Point(0, 1e5 + 40.8, 0),  Color(.75, .75, .25),     Color (0, 0, 0),0, Color(), Color(),0), // ceiling
	Sphere(0,  Point(24, 24.3, -50),  Color(),   Color(0, 8000, 8000),0, Color(), Color(),0), // light source 1
	Sphere(0,  Point(-24, 24.3, -50),  Color(),   Color(8000, 0, 8000),0, Color(), Color(),0), // light source 2
	Sphere(0, Point(0, -24.3, -30), Color(),   Color(8000, 8000, 0),0, Color(), Color(),0) //light source 3
	*/
	

	//Escena: radio, posicion, color 
	Sphere(1e5,  Point(-1e5 - 49, 0, 0),  Color(.5, .5, .5), Color(), 0, Color(), Color(), 0), // pared izq
	Sphere(1e5,  Point(1e5 + 49, 0, 0),    Color(.5, .5, .5), Color(), 0, Color(), Color(), 0), // pared der
	Sphere(1e5,  Point(0, 0, -1e5 - 81.6), Color(.5, .5, .5), Color(), 0, Color(), Color(), 0), // pared detras
	Sphere(1e5,  Point(0, -1e5 - 40.8, 0),Color(.5, .5, .5), Color(), 0, Color(), Color(), 0), // suelo
	Sphere(1e5,  Point(0, 1e5 + 40.8, 0), Color(.5, .5, .5), Color(), 0, Color(), Color(), 0), // techo
	Sphere(0, Point(0,24.3,-10), Color(1, 1, 1), Color(8000,8000,0),0, Color(), Color(),0), // esfera arriba
	//Sphere(16.5, Point(-23, -24.3, -34.6), Color(), Color(), 0, Color(), Color(), 0),
	//Sphere(16.5, Point(-23, -24.3, -34.6), Color(), Color(), 1, Color(1.66058, 0.88143, 0.521467), Color(9.2282, 6.27077, 4.83803), 0.03), // esfera abajo-izq
	//Sphere(16.5, Point(23, -24.3, -3.6),   Color(), Color(), 1, Color(0.143245, 0.377423, 1.43919), Color(3.98479, 2.3847, 1.60434), 0.3), // esfera abajo-der
	//Sphere(16.5, Point(-23, -24.3, -20.6), Color(.75, .75, .25), Color(), 3, Color(), Color(), 0),
	Sphere(16.5, Point(23, -24.3, -3.6), Color(.70, .3, 0), Color(), 0, Color(), Color(), 0),
	Sphere(0, Point(-23, 0, -10.6), Color(1, 1, 1), Color(0,8000,8000),0, Color(), Color(), 0),
	Sphere(0, Point(23, 24.3, -50), Color(1, 1, 1), Color(8000,0,8000),0, Color(), Color(), 0)
	
};
//prototipos
double transmitance(Point x1, Point x2, double sigma_t);
double multipleT(Point x1, Point x2 , double sigma_t);

inline void coordinateSystem( Vector &n, Vector &s, Vector &t) {
	if (std::abs(n.x) > std::abs(n.y)) {
	double invLen = 1.0 / std::sqrt(n.x * n.x + n.z * n.z);
	t = Vector(n.z * invLen, 0.0f, -n.x * invLen);
}	else {
	double invLen = 1.0 / std::sqrt(n.y * n.y + n.z * n.z);
	t = Vector(0.0f, n.z * invLen, -n.y * invLen);
	}
	s = t%n;
}

inline void coordinateTraspose(Vector n, Vector &w){ //pasa w direccion local con base a la normal
	Vector wlocal = w;
	Vector s,t;
	coordinateSystem(n, s, t);
	Vector ninv, sinv, tinv;
	sinv = Vector(s.x, t.x, n.x); 
	tinv = Vector(s.y, t.y, n.y); 
	ninv = Vector(s.z, t.z, n.z);
	w = sinv*wlocal.x+tinv*wlocal.y+ninv*wlocal.z; //vector en local 
}


// limita el valor de x a [0,1]
inline double clamp(const double x) { 
	if(x < 0.0)
		return 0.0;
	else if(x > 1.0)
		return 1.0;
	return x;
}

// convierte un valor de color en [0,1] a un entero en [0,255]
inline int toDisplayValue(const double x) {
	return int( pow( clamp(x), 1.0/2.2 ) * 255 + .5); 
}


// PROYECTO 1
// calcular la intersección del rayo r con todas las esferas
// regresar true si hubo una intersección, falso de otro modo
// almacenar en t la distancia sobre el rayo en que sucede la interseccion
// almacenar en id el indice de spheres[] de la esfera cuya interseccion es mas cercana
inline bool intersect(const Ray &r, double &t, int &id) {
	double tmin = __DBL_MAX__;
	double tact; //distancia actual a la esfera
	int contact=0;

	int elements = sizeof(spheres)/sizeof(spheres[0]);

	for(int i=0; i<elements; i++){
		tact=spheres[i].intersect(r);
		if(tact>0 && abs(tact)>0.0001){ //abs es necesario para eliminar intersecciones con el interior de las esfera dadas por errores de precision
			contact++; //se cambia el estado a que el contacto sucedio
			if(tact<tmin){
				tmin=tact;
				id=i;
				}	}	}
	if(contact>0){
		t=tmin;
		return true; 
	} 
	else{
		t=0;
		return false;
	} 
}

//intersect v2, devulve la primera y segunda interseccion de la esfera mas cercana
inline bool intersectV2(const Ray &r, double &t1, double &t2, int &id) {
	double tmin = __DBL_MAX__;
	double tact1, tact2; //distancia actual a la esfera
	int contact=0;

	int elements = sizeof(spheres)/sizeof(spheres[0]);

	for(int i=0; i<elements; i++){
		spheres[i].intersectVPT(r, tact1, tact2);
		if(tact1>0 && abs(tact1)>0.0001){ //abs es necesario para eliminar intersecciones con el interior de las esfera dadas por errores de precision
			contact++; //se cambia el estado a que el contacto sucedio
			if(tact1<tmin){
				tmin=tact1;
				t1=tact1;
				t2=tact2;
				id=i;
				}	}	}
	if(contact>0){
		return true; 
	} 
	else{
		t1=0;
		t2=0;
		return false;
	} 
}

//version de intersect para vpt, calcula la interseccion mas cercana ignorado la esferas de tipo==3
inline bool intersectVPT(const Ray &r, double &t, int &id) {
	double tmin = __DBL_MAX__;
	double tact; //distancia actual a la esfera
	int contact=0;

	int elements = sizeof(spheres)/sizeof(spheres[0]);

	for(int i=0; i<elements; i++){
		if(spheres[i].material!=3){ //ignorar esferas volumetricas, las podemos "atravesar"
			tact=spheres[i].intersect(r);
			if(tact>0 && abs(tact)>0.0001){ //abs es necesario para eliminar intersecciones con el interior de las esfera dadas por errores de precision
				contact++; //se cambia el estado a que el contacto sucedio
				if(tact<tmin){
					tmin=tact;
					id=i;
					}	}	}
	}
	if(contact>0){
		t=tmin;
		return true; 
	} 
	else{
		t=0;
		return false;
	} 
	
}



//devuelve wi con muestreo hemisferico uniforme
inline Vector uniformHemispheric(Vector n){ 
		
		double theta = acos(erand48(seed));
		double phi = 2*M_PI*(erand48(seed));

		//es direccion por lo tanto r = 1 (direcciones locales)
		double x1 = sin(theta)*cos(phi);
		double y1 = sin(theta)*sin(phi);
		double z1 = cos(theta);

		Vector s1,t1 = Vector(); //la normal ya esta dada por el punto x
		coordinateSystem(n, s1, t1);
		Vector wi = s1*x1+t1*y1+n*z1;

		wi.normalize();
		return wi;
}

//devuelve wi con muestreo esferico uniforme
inline Vector uniformSpheric(){ 		
		double theta = acos(1-2*erand48(seed));
		double phi = 2*M_PI*erand48(seed);

		//es direccion por lo tanto r = 1 (direcciones locales)
		double x1 = sin(theta)*cos(phi);
		double y1 = sin(theta)*sin(phi);
		double z1 = cos(theta);

		Vector wi = Vector(x1,y1,z1);

		wi.normalize();
		return wi;
}

//muestreo de coseno hemisferico
inline Vector cosineHemispheric(Vector n){ //n es z para la direccion local muestreada
		double theta = acos(sqrt(1-erand48(seed)));
		double phi = 2*M_PI*erand48(seed);

		//es direccion por lo tanto r = 1 (direcciones locales)
		double x1 = sin(theta)*cos(phi);
		double y1 = sin(theta)*sin(phi);
		double z1 = cos(theta);

		Vector s1,t1 = Vector(); //la normal ya esta dada por el punto x
		coordinateSystem(n, s1, t1);
		Vector wi = s1*x1+t1*y1+n*z1;

		wi.normalize();
		return wi;
}

inline Vector solidAngle(Vector wc, double costheta_max){
	double e0 = erand48(seed);
	double theta = acos((1-e0)+e0*costheta_max);
	double phi = 2*M_PI*erand48(seed);

	//convertir el angulo a coordenadas locales
	double x1 = sin(theta)*cos(phi);
	double y1 = sin(theta)*sin(phi);
	double z1 = cos(theta);

	//trasladar a coordenadas globales
	Vector s1,t1 = Vector(); //la normal ya esta dada por el punto x
	coordinateSystem(wc, s1, t1);
	Vector wi = s1*x1+t1*y1+wc*z1;

	wi.normalize();
	return wi;
}

//devuelve la probabilidad para muestreo de angulo solido
inline double solidAngleProb(double costheta_max){ 
	return 1/(2*M_PI*(1-costheta_max));
} 



//devuelve la probabilidad para muestreo de coseno hemisferico
inline double hemiCosineProb(double cosine){
	return cosine*1/M_PI;
}

//resuelve la visibilidad para dos puntos
bool visibility(Point light, Point x){
	Vector lx = light-x; //vector con direccion a la fuente
	double distance = sqrt(lx.dot(lx)); //distancia a la fuente
	lx.normalize();
	lx=lx*-1; //invierte la direccion
	Ray r2 = Ray(light, lx);
	int id=0; 
	double t;
	intersect(r2, t, id);
	if(t>distance || t==0)
	{
		return true; //devolver radiancia
	}
	else return false;
}
//visibilidad para vpt
bool visibilityVPT(Point light, Point x){
	Vector lx = light-x; //vector con direccion a la fuente
	double distance = sqrt(lx.dot(lx)); //distancia a la fuente
	lx.normalize();
	lx=lx*-1; //invierte la direccion
	Ray r2 = Ray(light, lx);
	int id=0;
	double t;
	intersectVPT(r2, t, id);
	if(t>distance || t==0)
	{
		return true; //devolver radiancia
	}
	else return false;
}

//resuelve la visibilidad de forma implicita (no sirve para muestreo de luz)
inline Color rayTracer(Point x, Vector wi, int &sourceid){ 
	Ray r1 = Ray(x, wi);
	double t;
	int id = 0;
	if(!intersect(r1, t, id)) return Color();
	sourceid = id;
	const Sphere &source = spheres[id];
	return source.radiance;
	}

//muestreo uniforme
inline Color uniform(Vector n, Point x, Color BDRF, Vector &aux, int &idsource){ //aux obtiene de regreso la direccion muestreada
	int sourceid;
	Color L, Le;
	Vector wi;
	wi = cosineHemispheric(n);
	wi.normalize();
	Le = rayTracer(x, wi, sourceid);
	idsource = sourceid;
	L = L + Le.mult(BDRF*(1/M_PI))*n.dot(wi)*(1/hemiCosineProb(n.dot(wi)));
	aux = wi;
	return L;
}

//calcula el fresnel para solo un espectro
double fresnelSpectre(double cosine, double sine, double eta, double kappa){
	double a2b2 = sqrt(((eta)*(eta)-(kappa)*(kappa)-sine*sine)*((eta)*(eta)-(kappa)*(kappa)-sine*sine)+4*(eta)*(eta)*(kappa)*(kappa));
	double a = sqrt(0.5*(a2b2+eta*eta-kappa*kappa-sine*sine));
	double parallel, perpendicular;
	perpendicular = (a2b2+cosine*cosine-2*a*cosine)/(a2b2+cosine*cosine+2*a*cosine);
	parallel = perpendicular*(a2b2*cosine*cosine+sine*sine*sine*sine-2*a*cosine*sine*sine)/(a2b2*cosine*cosine+sine*sine*sine*sine+2*a*cosine*sine*sine);
	return 0.5*(parallel+perpendicular);
}

//calcula la naturaleza de la luz reflejada para modelos con reflexion
Color fresnel(double cosine_wh, Vector eta, Vector kappa){ //el argumento es el coseno del angulo con respecto a wh
	//se debe calcular para cada espectro en eta y kappa (x,y,z)
	double sine_wh = sqrt(1-cosine_wh*cosine_wh);
	Color fwh;
	fwh.x = fresnelSpectre(cosine_wh, sine_wh, eta.x, kappa.x);
	fwh.y = fresnelSpectre(cosine_wh, sine_wh, eta.y, kappa.y);
	fwh.z = fresnelSpectre(cosine_wh, sine_wh, eta.z, kappa.z);
	return fwh;
}



//calcula la distribucion NDF, usa el angulo theta_h, cos(theta_h)=wh.dot(n);
double NDF(double cosine, double alpha){
	double fac1, fac2, tang, sine;
	if(cosine>=0){
		sine = sqrt(1-cosine*cosine);
		fac1 = M_PI*alpha*alpha*cosine*cosine*cosine*cosine;
		tang = sine/cosine;
		fac2 = exp((-1*tang*tang)/(alpha*alpha));
		return (1/fac1)*fac2;

	}
	else return 0;	
}

double Gn(Vector n, Vector wv, Vector wh, double alpha){
	double sin = sqrt(1-n.dot(wv)*n.dot(wv));
	double tan = sin/(n.dot(wv)); //tan = sen/cos
	double a = 1/(alpha*tan);
	double num, den;
	if(((wv.dot(wh))/(wv.dot(n)))>0){
		if(a<1.6){
			num = 3.535*a+2.181*a*a;
			den = 1+2.276*a+2.577*a*a;
			return num/den;
		}
		else return 1;
	}
	else return 0;
}

double G_smith(Vector n, Vector wi, Vector wo, Vector wh, double alpha){
	double g1, g2;
	g1 = Gn(n, wi, wh, alpha);
	g2 = Gn(n, wo, wh, alpha);
	return g1*g2;
}

//obtener direccion wh
inline Vector vectorFacet(double alpha){
	Vector wh;
	double theta = atan(sqrt(-alpha*alpha*log(1-erand48(seed))));
	double phi = 2*M_PI*erand48(seed);

	double x1 = sin(theta)*cos(phi);
	double y1 = sin(theta)*sin(phi);
	double z1 = cos(theta);

	wh = Vector(x1, y1, z1);
	wh.normalize();

	return wh;
}
//calcula la probabilidad para el modelo microfacet
double microFacetProb(Vector wo, Vector wh, double alpha, Vector n){//n debe estar en mismo sistema que wh y wo
	//wo en local
	double num, den;
	num = wh.dot(n);
	den = 4 * abs(wo.dot(wh));
	return NDF(wh.dot(n), alpha)*num/den;
}

//evaluacion de la BDRF para el modelo microfacet
inline Color frMicroFacet(Color eta, Color kappa, Vector wi, Vector wh, Vector wo, double alpha, Vector n){
	//Vector n = (0,0,1);
	//es necesario que los vectores cumplan estar normalizados, ser direcciones salientes y en direccion local
	double den =(4*abs(n.dot(wi))*abs(n.dot(wo)));
	return fresnel(wi.dot(wh), eta, kappa)*NDF(n.dot(wh), alpha)*G_smith(n, wi, wo, wh, alpha)*(1/den);
}

//muestreo de materiales microfacet (el modelo emplea coordenadas locales)
inline Color microfacet(Point x, Vector wo, Vector wh, Vector n, Sphere &obj, double alpha, int &idsource){ 
	Color Le, fr;
	Vector s, t;
	Vector nlocal = Vector(0,0,1);
	int sourceid;
	wo = wo*(-1); //se invierte para respetar la convencion de direcciones salientes
	//wo debe pasarse a local para que todos los vectores esten en el mismo marco de referencia n=(0,0,1)
	coordinateTraspose(n, wo);
	wo.normalize();
	Vector wi = wo*(-1)+wh*2*(wh.dot(wo)); //reflexion especular, direccion reflejada
	wi.normalize();

	coordinateSystem(n, s, t);
	Vector wiglobal = s*wi.x+t*wi.y+n*wi.z;
	wiglobal.normalize();
	//lanzar un rayo direccion wi, debe globalizarse para resolver visibilidad
	Le = rayTracer(x, wiglobal, sourceid);
	idsource = sourceid;
	fr = frMicroFacet(obj.eta, obj.kappa, wi, wh, wo, alpha, nlocal);

	return Le.mult(fr)*Vector(0,0,1).dot(wi)*(1/microFacetProb(wo, wh, alpha, nlocal));
}


//L muestreo de area
inline Color areaLight(Point center, double radio, Point x, Color radiance, const Sphere &obj, Vector n, Vector wo, double alpha){
	Vector xl;
	Point light;
	Color le, fr, L;
	Vector aux = uniformSpheric(); //respecto a z
	light = center + aux*radio; //punto muestreado
	xl = x-light;
	if(aux.dot(xl.normalize())<0)
		return 0;
	if(visibility(light, x))
		le = radiance; //radiancia de la fuente de luz
	else 
		return 0;
	Vector wilocal = xl*-1;
	Vector wolocal = wo*-1; 

	coordinateTraspose(n, wolocal);
	coordinateTraspose(n, wilocal);
	wolocal.normalize();
	wilocal.normalize();
		
	Vector wh = wilocal+wolocal;
	wh.normalize();

	if(obj.material==0)	fr = obj.c*(1/M_PI);
	else{
		fr = frMicroFacet(obj.eta, obj.kappa, wilocal, wh, wolocal, 0.3, Vector(0,0,1));
	}
	double prob = (light-x).dot(light-x)/(4*M_PI*radio*radio*aux.dot(xl)); 
	L = le.mult(fr)*n.dot((light-x).normalize())*(1/prob);
	return L;
}

//devuelve color calculado para muestreo de area
inline Color muestreoArea(const Sphere &source, const Sphere &obj, Point x, Vector n, Vector wray, double alpha){
	Color L = areaLight(source.p, source.r, x, source.radiance, obj, n, wray, 0.3);
	return L;
}

//L de Solid Angle
inline Color solidAngle(Vector n, Vector cx, Vector wray, double costheta_max, Point x, int indice, const Sphere &obj, Vector &aux, double alpha){ 
	Color L , Le;
	Vector wi;
	Color fr;
	Vector wolocal = wray*-1;
	Vector wilocal;
	wi = solidAngle(cx, costheta_max); //wi guarda la direccion muestreada
	aux = wi;
	Ray r1 = Ray(x, wi); //lanzar el rayo desde x con direccion wi
	wilocal = wi;

	coordinateTraspose(n, wilocal);
	coordinateTraspose(n, wolocal);
	
	wilocal.normalize();//el cambio en el sistema de coordenadas altera la magnitud del vector
	wolocal.normalize();

	Vector wh = (wilocal+wolocal);
	wh.normalize();

	//para aluminio eta =[1.66058, 0.88143, 0.521467] kappa = [9.2282, 6.27077, 4.83803]
	//para oro eta = [0.143245, 0.377423, 1.43919] kappa = [3.98479, 2.3847, 1.60434]
	if(obj.material==0){
		fr = obj.c*(1/M_PI);	
		}	
	else if(obj.material==2){
		//la evaluacion de un dielectrico suave siempre dara 0 en muestreo de luz, no es factible dar con la direccion correcta
		fr = Color(0,0,0);
	}
	else fr = frMicroFacet(obj.eta, obj.kappa, wilocal, wh, wolocal, alpha, Vector(0,0,1));
	double t;
	int id = 0; 
	intersect(r1, t, id);
	const Sphere &source = spheres[id];

	if(indice==id) Le=source.radiance; //nos aseguramos que exista visibilidad entre la fuente y no otras fuentes de luz
	else Le = Color();
	//funcion L
	L = Le.mult(fr)*n.dot(wi)*(1/solidAngleProb(costheta_max));

	return L;
}


//L de Solid Angle en ray marching
inline Color solidAngleMarching(Vector n, Vector cx, Vector wray, double costheta_max, Point x, int indice, const Sphere &obj, Vector &aux, double alpha){ 
	Color L , Le;
	Vector wi;
	Color fr;
	Vector wolocal = wray*-1;
	Vector wilocal;
	wi = solidAngle(cx, costheta_max); //wi guarda la direccion muestreada
	aux = wi;
	Ray r1 = Ray(x, wi); //lanzar el rayo desde x con direccion wi
	wilocal = wi;

	coordinateTraspose(n, wilocal);
	coordinateTraspose(n, wolocal);
	
	wilocal.normalize();//el cambio en el sistema de coordenadas altera la magnitud del vector
	wolocal.normalize();

	Vector wh = (wilocal+wolocal);
	wh.normalize();

	//para aluminio eta =[1.66058, 0.88143, 0.521467] kappa = [9.2282, 6.27077, 4.83803]
	//para oro eta = [0.143245, 0.377423, 1.43919] kappa = [3.98479, 2.3847, 1.60434]
	if(obj.material==0){
		fr = obj.c*(1/M_PI);	
		}	
	else if(obj.material==2){
		//la evaluacion de un dielectrico suave siempre dara 0 en muestreo de luz, no es factible dar con la direccion correcta
		fr = Color(0,0,0);
	}
	else fr = frMicroFacet(obj.eta, obj.kappa, wilocal, wh, wolocal, alpha, Vector(0,0,1));
	double t;
	int id = 0; 
	intersect(r1, t, id);
	const Sphere &source = spheres[id];

	if(indice==id) Le=source.radiance; //nos aseguramos que exista visibilidad entre la fuente y no otras fuentes de luz
	else Le = Color();
	//funcion L
	L = Le.mult(fr)*n.dot(wi)*(1/solidAngleProb(costheta_max));
	

	return Le;
}

//calcula la naturaleza de la luz reflejada para modelos con reflexion y transmision
/*
etai corresponderia al medio
etat corresponderia al material
*/
inline double fresnelDie(double etai, double etat, double cosinet, double cosinei){
    double parallel, perpendicular;
    parallel = ((etat*cosinei-etai*cosinet)/(etat*cosinei+etai*cosinet))*((etat*cosinei-etai*cosinet)/(etat*cosinei+etai*cosinet));
    perpendicular = ((etai*cosinei-etat*cosinet)/(etai*cosinei+etat*cosinet))*((etai*cosinei-etat*cosinet)/(etai*cosinei+etat*cosinet));
    return 0.5*(parallel+perpendicular);
}
 
 
 
//muestreo dielectrico reflexion
inline Vector reflexDielectric(Vector wi, Vector n){ //wi debe entrar como saliente y no incidente wi = r.d*-1
    //direccion reflejada wr
    return wi*-1+n*(n.dot(wi))*2;
}
 
//muestreo dielectrico transmision
inline Vector refraxDielectric(double etai, double etat, Vector wi, Vector n){ //wo sustituye a wi dado que la direccion incidente es la de observacion
    //falta convertir a direcciones locales y obtener la wt global de cualquier form
	//direccion refractada wt
	Vector wilocal = wi; 
	Vector wtglobal;
	coordinateTraspose(n, wilocal);
	//std::cout<<acos((wilocal).dot(Vector(0,0,1)))<<std::endl;
    double ratio = etat/etai*-1;
    double cosinei = (wi).dot(n);
	double invratio = etai/etat;
    double cosinet = sqrt(1-invratio*invratio*(1-cosinei*cosinei))-1; //cosine de la nueva direccion
    Vector wtlocal = Vector(wilocal.x*ratio, wilocal.y*ratio, cosinet);
	Vector s, t;

	coordinateSystem(n, s, t);
	//Vector global 
	wtglobal = s*wtlocal.x+t*wtlocal.y+n*wtlocal.z;
	return wtglobal;
}


//muestreo de dielectrico suave
inline Color softDielectric(double etat, double etai, Vector wi, Vector n, Point x, int &idsource){ //wi sera la direccion incidente en este modelo (r.d)
    Color Ld;
    int sourceid;
    //direcciones de transmision para calcular Fresnel
    Vector wt = refraxDielectric(etai, etat, wi, n);
    wt.normalize();
    double F = fresnelDie(etai, etat, n.dot(wt), n.dot(wi));
    //muestreo
    if(erand48(seed)<F){
        //calcula reflexion
        Vector wr = reflexDielectric(wi, n);
        wr.normalize();
        //con reflexion
        //la probailidad de muestreo es F, F esta en el numerador ... 1
        Ld = rayTracer(x, wr, sourceid)*(1/abs(n.dot(wr)));
 
    }
    else{
        //con transmision
        double ratio = etat/etai;
        //la probabilidad de muestreo es 1-F, 1-F esta en el denominador ... 1
        Ld = rayTracer(x, wt, sourceid)*(1/abs(n.dot(wt)))*ratio*ratio;
		
    }
	idsource = sourceid;
    return Ld;
}


inline double cosinethetaMax(int sourceid, Point x){
	double radio = spheres[sourceid].r;
	Vector cx = spheres[sourceid].p-x;
	double normcx = sqrt(cx.dot(cx));
	cx.normalize();
	double costheta_max = sqrt(1-(radio/normcx)*(radio/normcx));
	return costheta_max;
}

//muestreo de angulo solido
inline Color muestreoSA(Sphere &source, Point x, int indice, const Sphere &obj, Vector n, Vector wray, Vector &aux, double &costhetaMax, double alpha){
	Color L; 
	Vector cx = source.p-x;
	double normcx = sqrt(cx.dot(cx));
	cx = cx*(1/normcx);
	double costheta_max = sqrt(1-(source.r/normcx)*(source.r/normcx));
	costhetaMax = costheta_max;
	L = solidAngle(n, cx, wray, costheta_max, x, indice, obj, aux, alpha);
	return L;
}

//muestreo de angulo solido
inline Color marchingSA(Sphere &source, Point x, int indice, const Sphere &obj, Vector n, Vector wray, Vector &aux, double &costhetaMax, double alpha){
	Color L; 
	Vector cx = source.p-x;
	double normcx = sqrt(cx.dot(cx));
	cx = cx*(1/normcx);
	double costheta_max = sqrt(1-(source.r/normcx)*(source.r/normcx));
	costhetaMax = costheta_max;
	L = solidAngle(n, cx, wray, costheta_max, x, indice, obj, aux, alpha);
	return L;
}



//calcula la contribucion de una luz puntual
inline Color pLight(const Sphere &obj, Point x, Vector n, Vector wray, Color I, Point light, double alpha){ 
	Color Le;
	Color fr;
	if(visibility(light, x))
	{
		Le = I*(1/((light-x).dot(light-x))); //devolver radiancia
	}
	else if(visibilityVPT(light, x)) 
	{
		Le = I*(1/((light-x).dot(light-x))); //devolver radiancia
		Le = Le*multipleT(x, light,0.05+0.009);
	}
	else Le = Color(); //no hay visibilidad
	//modelo microfacet
	Vector wi = light-x;
	wi.normalize();
	Vector wo = wray*-1;
	coordinateTraspose(n, wo);
	coordinateTraspose(n, wi);
	wi.normalize();
	wo.normalize();
	Vector wh = wi+wo;
	wh.normalize();
	if(obj.material==1){
		fr=frMicroFacet(obj.eta, obj.kappa, wi, wh, wo, alpha, Vector(0,0,1));
	}
	else fr = obj.c*(1/M_PI);
	Color L = Le.mult(fr)*n.dot((light-x).normalize());
	return L;
}

//funcion no utilizada aun 
inline Point intersect(const Ray &r, const Vector &n){
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return {};	// el rayo no intersecto objeto, return Vector() == negro
	Point x = r.o+r.d*t;
	return x;
}

//calcula la heuristica de potencia B=2
inline double powerHeuristics(double fpdf, double gpdf){
	double f2 = fpdf*fpdf;
	double g2 = gpdf*gpdf;
	return f2/(f2+g2);
}

//devuelve la integral para multiple importance sampling
inline Color MIS(Sphere &obj, Point x, Vector n, Vector wray, double alpha){ //for each ligth source
	Color f, g, montecarlo = Color();
	Vector wiLight, wiBDRF, wo;
	double wf, wg, fpdf, gpdf, costhetaMax;
	int sourceid, sourceid2=0;
	wo = wray*-1;

//se muestrea each light source
	int a = sizeof(spheres);
	int b = sizeof(spheres[0]);
	int size = a/b; //cantidad de esferas en la escena
	for(int light = 0; light<size; light++) {
		if(spheres[light].r>0 && spheres[light].radiance.x>0){
			f = muestreoSA(spheres[light], x, light, obj, n, wray, wiLight, costhetaMax, alpha);
			fpdf = solidAngleProb(costhetaMax);
			if(obj.material==0) gpdf = hemiCosineProb(n.dot(wiLight));
			else if(obj.material==2){
			//calcular el fresnel para la direccion muestreada por luz prob = F en reflexion
				Vector wt = refraxDielectric(1.0, 1.5, wo, n);
				wt.normalize();
				gpdf = fresnelDie(1.0, 1.5, n.dot(wt), n.dot(wo));
				if(erand48(seed)>gpdf){
				gpdf=1-gpdf;
				}	
	}
			else{
				Vector wh = wiLight+wo; //global
				wh.normalize();
				gpdf = microFacetProb(wo, wh, alpha, n);
	} 
			wf = powerHeuristics(fpdf, gpdf);
			montecarlo = montecarlo + f*wf;

		}
		
	}
	
//se muestrea BDSF (es agnostico a la fuente, puede darle a cualquiera)
	if(obj.material==0){
		g = uniform(n, x, obj.c, wiBDRF, sourceid);
		gpdf = hemiCosineProb(n.dot(wiBDRF));
		//para obtener el costhetaMax apropiado, considerar la fuente a la que la direccion apunta
		if(g.x>0 && g.y>0 && g.z>0){ costhetaMax = cosinethetaMax(sourceid, x);
			fpdf = solidAngleProb(costhetaMax);
			wg = powerHeuristics(gpdf, fpdf);
		}	
		else wg = 0;
		
		
	}
	else if(obj.material==2){
		//calcula iluminacion para dielectrico suave
		g = softDielectric(1.5, 1.0, wo, n, x, sourceid);
		if(g.x>0 && g.y>0 && g.z>0){ costhetaMax = cosinethetaMax(sourceid, x);
		fpdf = solidAngleProb(costhetaMax);
		wg = powerHeuristics(gpdf, fpdf);
		}
		else wg = 0;
	}
	else{
		Vector wh = vectorFacet(alpha);
		coordinateTraspose(n, wo);
		wo.normalize();
		g = microfacet(x, wray, wh, n, obj, alpha, sourceid2);
		gpdf = microFacetProb(wo, wh, alpha, Vector(0,0,1));//local
		//para obtener el costhetaMax apropiado, considerar la fuente a la que la direccion apunta
		if(g.x>0)	costhetaMax = cosinethetaMax(sourceid2, x);
		fpdf = solidAngleProb(costhetaMax);
		wg = powerHeuristics(gpdf, fpdf);
	} 

	//MIS
	
	montecarlo = montecarlo + g*wg;

	return montecarlo;
}


//path tracing explicito recursivo con MIS y ruleta rusa constante
inline Color explicitPathRecursive(const Ray &r, int bounce){
	double t;
	int id = 0;

	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() == negro
  

	if(spheres[id].radiance.x > 0)
		return Color();

	Point x = r.o + r.d*t;

	Vector n = (x-spheres[id].p);  
	n.normalize();
	
	Vector wi, wh;
	Vector wo = r.d * -1;
	Color  fs, Ld, Lind;
	double cosine, prob;
	
	//calcular Ld for each light 
	Ld = MIS(spheres[id], x, n, r.d, 0.001); //multiple importance sampling para iluminacion directa

	//ruleta rusa constante
	
	double q = 0.1;
	double continueprob = 1.0 - q;
	if(erand48(seed) < q)
		return Ld; // no continuar el camino (regresa 0 en la integral)
	//muestreo de BDSF para obtener Lind
	if(spheres[id].material==0){
		wi = cosineHemispheric(n);
		fs = spheres[id].c*(1/M_PI);
		prob = hemiCosineProb(n.dot(wi));
	}
	else{
		double alpha = 0.001;
		Vector wh = vectorFacet(alpha); //local
		Vector s, t;
		coordinateSystem(n, s, t);
		wh = s*wh.x+t*wh.y+n*wh.z; //global
		wi = wo*(-1)+wh*2*(wh.dot(wo));
		fs = frMicroFacet(spheres[id].eta, spheres[id].kappa, wi, wh, wo, alpha, n);
		prob = microFacetProb(wo, wh, alpha, n);
	}

	
	Ray recursiveRay = Ray(x, wi);
	cosine = n.dot(wi);
	bounce++;
	Lind = fs.mult(explicitPathRecursive(recursiveRay, bounce))*abs(cosine)*(1/(prob*continueprob));

	return (Ld + Lind);	
	//return explicitPathRecursive(recursiveRay, bounce, Ld, fs, cosine, prob)
}

//calcula la fs y la nueva direccion para un camino en Path tracing. Lind
inline Color BDSF(Vector &aux, Vector wray, Vector n,  double &prob, int id){
	Vector wi;
	Color fs1;
	Vector wo = wray*-1;
	if(spheres[id].material==0){
		wi = cosineHemispheric(n);
		fs1 = spheres[id].c*(1/M_PI);
		prob = hemiCosineProb(n.dot(wi));
		aux = wi;
	}
	else if(spheres[id].material == 2){
		Vector wt = refraxDielectric(1.0, 1.5, wo, n);
		wt.normalize();
		double F = fresnelDie(1.0, 1.5, n.dot(wt), n.dot(wo));
		if(erand48(seed)<F){
			//se obtiene reflexion
			wi = reflexDielectric(wo, n);
			wi.normalize();
			fs1=Color(1,1,1)*(1/n.dot(wi))*(F);
			
			prob = F;
		}
		else{
			wi = wt;
			fs1=Color(1,1,1)*(1/n.dot(wi))*(1-F)*(1.5)*1.5;

			prob = 1-F;

		}
		aux = wi;
	}
	else if(spheres[id].material == 1){
		double alpha = spheres[id].alpha;
		Vector wh = vectorFacet(alpha); //local
		Vector s, t;
		coordinateSystem(n, s, t);
		wh = s*wh.x+t*wh.y+n*wh.z; //global
		wi = wo*(-1)+wh*2*(wh.dot(wo));
		fs1 = frMicroFacet(spheres[id].eta, spheres[id].kappa, wi, wh, wo, alpha, n);
		prob = microFacetProb(wo, wh, alpha, n);
		aux = wi;
	}
	return fs1;
}

inline Color explicitPath(const Ray &r){
	double t;
	int id = 0;

	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() == negro
  
	const Sphere &obj = spheres[id];
	

	if(obj.radiance.x > 0)
		return obj.radiance;
	else return explicitPathRecursive(r, 0);
}

//recursividad de cola, se puede convertir a iteracion facilmente
inline Color tailExplicitPath(const Ray &r, int bounces, Color Accum, Color fs, double factor){
	double t;
	int id = 0;

	if (!intersect(r, t, id))
		return Accum;	// el rayo no intersecto objeto, return Vector() == negro
  

	if(spheres[id].radiance.x > 0)
		return Accum;

	Point x = r.o + r.d*t;

	Vector n = (x-spheres[id].p);  
	n.normalize();
	
	Vector wi, wh;
	Vector wo = r.d * -1;
	Color  fs1, Ld, Lind;
	double cosine, prob;
	
	//calcular Ld for each light 
	Ld = MIS(spheres[id], x, n, r.d, 0.001); //multiple importance sampling para iluminacion directa

	//ruleta rusa constante
	
	double q = 0.1;
	double continueprob = 1.0 - q;
	if(erand48(seed) < q)
		return Accum+fs.mult(Ld)*factor; // no continuar el camino (regresa 0 en la integral)
	//muestreo de BDSF para obtener Lind
	fs1 = BDSF(wi, r.d, n, prob, id);
	Ray recursiveRay = Ray(x, wi);
	cosine = n.dot(wi);
	//se añade el color directo
	Accum = Accum + fs.mult(Ld)*factor;
	fs = fs.mult(fs1);
	return tailExplicitPath(recursiveRay, bounces++, Accum, fs, factor*abs(cosine)*(1/(prob*continueprob)));
}

//Path tracer explicito iterativo con MIS y ruleta rusa 
inline Color iterativePathTracer(Ray r){
	Color Accum = Color();
	Color fs = Vector(1,1,1);
	Color Ld, fs1 = Color();
	Vector wi;
	int bounces = 0;
	double factor = 1;
	double q = 0.1;
	double continueprob = 1.0 - q;
	double prob;
	int i = 0;
	while(true){ //el ciclo se interrumpe en puntos especificos, se puede agregar un contador para limitar los caminos
		double t;
		int id = 0;
		if (!intersect(r, t, id))
			break;	// el rayo no intersecto objeto, return Vector() == negro
  

		if(spheres[id].radiance.x > 0){ //si se impacta una fuente de luz, regresa radiancia
			if(bounces<1) return spheres[id].radiance;
			break;
		}
			

		Point x = r.o + r.d*t;
		Vector n = (x-spheres[id].p);  
		n.normalize();
		//iluminacion directa
		//for each light source
		int a = sizeof(spheres);
		int b = sizeof(spheres[0]);
		int size = a/b; //cantidad de esferas en la escena

		for(int light = 0; light<size; light++) 
		{
			//resuelve las fuentes puntuales
			if(spheres[light].r==0) Ld = pLight(spheres[id], x, n, r.d, spheres[light].radiance, spheres[light].p, spheres[id].alpha) + Ld;
			
		}
		Ld = MIS(spheres[id], x, n, r.d, spheres[id].alpha) + Ld; //mis tiene su propio ciclo, por eso esta fuera del for
		//ruleta rusa
		if(erand48(seed) < q)
		{
			Accum = Accum + fs.mult(Ld)*factor;
			break;
		}
		//muestreo de BDSF para obtener Lind
		fs1 = BDSF(wi, r.d, n, prob, id);

		Ray newray = Ray(x, wi);
		r.o = newray.o;
		r.d = newray.d;
		double cosine = n.dot(wi); 
		Accum = Accum + fs.mult(Ld)*factor;
		fs = fs.mult(fs1);
		factor = factor*cosine*(1/(prob*continueprob));
		bounces++;
		Ld = Color();
	}
	return Accum;

}

inline Color implicitPath(const Ray &r, int bounces){
	double t;
	int id = 0;

	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() == negro
  
	const Sphere &obj = spheres[id];
	

	Point x = r.o + r.d*t;

	// determinar la dirección normal en el punto de interseccion
	Vector n = (x-obj.p);  // calcular x con el centro de la esfera como marco de referencia (radio)
	n.normalize();
	
	Vector wi, wh;
	Vector wo = r.d * -1;
	Color value, fs;
	double prob, cosine;

	//value = Le(x) light emitter con los que intersecta x 
	if(obj.radiance.x>0) 
		value =  obj.radiance; 
	
	if(bounces>5)
		return value;
	//sample BDSF
	if(obj.material==0){
		wi = cosineHemispheric(n);
		fs = obj.c*(1/M_PI);
		prob = hemiCosineProb(n.dot(wi));
	}
	else if(obj.material == 2){
		double etat=1.5;
		double etai=1.0;
		Vector wt = refraxDielectric(etai, etat, wo, n);
    	wt.normalize();
    	double F = fresnelDie(etai, etat, n.dot(wt), n.dot(wo));
		if(erand48(seed) < F)
		{
			wi = reflexDielectric(wo, n);
			wi.normalize();
			fs = (1/abs(n.dot(wi)));
			prob = 1.0;
		}	
		else{
			wi = wt;
			double ratio = etat/etai;
			fs = (1/abs(n.dot(wi)))*ratio*ratio;
			prob = 1.0;
		}
	
	}
	else if(obj.material == 1){
		double alpha = 0.3;
		Vector wh = vectorFacet(alpha); //local
		Vector s, t;
		coordinateSystem(n, s, t);
		wh = s*wh.x+t*wh.y+n*wh.z; //global
		wi = wo*(-1)+wh*2*(wh.dot(wo));
		fs = frMicroFacet(obj.eta, obj.kappa, wi, wh, wo, alpha, n);
		prob = microFacetProb(wo, wh, alpha, n);
	}
	wi.normalize();
	cosine = n.dot(wi);
	double q = 0.1;
	double continueprob = 1.0 - q;
	if(erand48(seed) < q)
		return value; // no continuar el camino (regresa 0 en la integral)

	Ray newray = Ray(x, wi);
	bounces++;
	value = (fs*abs(cosine)).mult(implicitPath(newray, bounces))*(1/(prob*continueprob)) + value;
	return value;

}


// Calcula el valor de color para el rayo dado
Color shade(const Ray &r) {
	
	double t;
	int id = 0;
	// determinar que esfera (id) y a que distancia (t) el rayo intersecta
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() == negro
  
	const Sphere &obj = spheres[id];
	
	// PROYECTO 1
	// determinar coordenadas del punto de interseccion
	Point x = r.o + r.d*t;

	// determinar la dirección normal en el punto de interseccion
	Vector n = (x-obj.p);  // calcular x con el centro de la esfera como marco de referencia (radio)
	n.normalize();

	if(obj.radiance.x>0) 
		return obj.radiance; //si un rayo impacta una fuente de luz se devuelve la radiancia directamente
	
	//muestreo de fuentes de luz
	
	int a = sizeof(spheres);
	int b = sizeof(spheres[0]);
	int size = a/b; //cantidad de esferas en la escena
	
	Color L_total, L = Color();

	for(int i=0; i<size; i++){
		if(spheres[i].r==0){
			L = pLight(obj,x ,n, r.d, spheres[i].radiance, spheres[i].p, 0.0003); //luces puntuales
		}	
		
	}
	L = MIS(spheres[id], x, n, r.d, 0.003)+L;	
	L_total = L_total+L;
	/*
	Muestreo no basado en fuentes de luz (resuelve la escena completa en una solo muestreo, no fuentes puntuales)
	*/	
	//Color L_uniform = uniform(64, n, x, obj.c*(1/M_PI));

	//luces puntuales
	/*
	Color pcolor = Color();
	pcolor = plight(obj, x, n, spheres[7].radiance, spheres[7].p); 
	*/

	return L_total;
}

//funciones para ray marching

double transmitance(Point x1, Point x2, double sigma_t){
	//obten la distancia entre dos puntos
	double d2 = (x2-x1).dot(x2-x1);
	double d = sqrt(d2);
	//devuelve la transmitancia
	return exp(-sigma_t*d);
}

double isotropicPhaseFunction(){
	//devuelve la funcion de fase isotropica
	return 1/(4*M_PI);
}

//funcion para muestreo explicito en ray marching con iluminacion global
Color rayMarching(const Ray &r, double sigma_t, double sigma_s, double steps, Point &x_new, int &idsource){
	
	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	idsource = id;
	Point x = r.o + r.d*t;
	//se guarda el punto de interseccion
	x_new = x;
	Color Li = Color();
	Color Lo = Color();
	//si el rayo intersecta una fuente de luz, Lo = radiancia * transmitancia
	if(spheres[id].radiance.x>0) {
		//como la fuente ya fue muestreada, no se toma en cuenta en el ray marching
		return Color();
	}
	//divide la distancia entre steps para obtener la distancia de cada segmento
	double step = t/steps;
	//para cada segmento se calcula la transmitancia y la funcion de fase
	for(int i = 0; i<steps; i++){
		Point xt = r.o + r.d*step*i;
		//calcular la transmitancia
		double T = transmitance(x, xt, sigma_t);
		//calcular la funcion de fase
		double phase = isotropicPhaseFunction();
		//calcular Ls aproximado
		//lanzar un shadow rays con angulo solido 
		//vector wc de xt a la fuente
		Vector wc = spheres[5].p-xt;
		double normcx = sqrt(wc.dot(wc));
		wc = wc*(1/normcx);
		double costheta_max = sqrt(1-(spheres[5].r/normcx)*(spheres[5].r/normcx));
		//vector wi sampleado con angulo solido
		Vector wi = solidAngle(wc, costheta_max);
		//rayo con origen en xt y direccion wi
		Ray shadowRay = Ray(xt, wi);
		//verificar la visibilidad con la fuente de luz
		//si el shadow ray intersecta en el mismo id que la fuente de luz, entonces hay visibilidad
		//en caso contrario no hay visibilidad
		double t2;
		int id2 = 0;
		intersect(shadowRay, t2, id2);
		//si hay visibilidad, la contribucion es la radiancia de la fuente de luz

		if(id2==5){//se tiene que modificar para que tome en cuenta varias fuentes (workaround)
			//obtener la radiancia de la fuente de luz
			Color Le = spheres[5].radiance;
			//calcular Ls
			//montecarlo de una sola muestra
			Color Ls = Le*(phase*transmitance(xt, spheres[5].p, sigma_t));
			//probabilidad de muestreo
			double prob = solidAngleProb(costheta_max);
			//calcular Ls aproximado
			Color Ls_aprox = Ls*(T*1/prob)*sigma_s*step;
			//acumular Ls aproximado
			Li = Li + Ls_aprox;
		}
		//en caso de no haber visibilidad
		else{
			Color Ls_aprox = Color();
			Li = Li + Ls_aprox;
		}

	}
		return Li;


}

//ray marching para volumenes, segmento o paso variable
Color rayMarchingGlobal(const Ray &r, double sigma_a,double sigma_s, double  segmentos){

	double sigma_t = sigma_a+sigma_s;
	
	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;
	Color Li = Color();
	Color Lo = Color();
	//si el rayo intersecta una fuente de luz, Lo = radiancia * transmitancia
	if(spheres[id].radiance.x>0) {
		return spheres[id].radiance*transmitance(r.o, x, sigma_t);
		//Lo = spheres[id].radiance*transmitance(r.o, x, sigma_t);
		
	}
	
	
	/*
	//calcular el color con angulo solido
	Vector aux = Vector();
	double costheta_max;
	Color Ld = muestreoSA(spheres[5], x, 5, spheres[id], n, r.d, aux, costheta_max, 0.001);
	//sumar Ld * transmitancia a Lo, solo si LO es 0
	if(Lo.x==0) Lo = Lo + Ld*transmitance(r.o, x, sigma_t);
	//si se desea aplicar multiple scattering, se debe aplicar ray marching despues de muestrear la direccion de angulo solido
	//sustituir en muestreo SA el montecarlo normal por ray marching
	//Le = emision de la fuente de luz * transmitancia + suma de reiman ... Le = Li
	//es decir luz entrante en el punto x = Li
	*/

	//obtener Le correspondiente a la fuente de luz
	//se debe iterar esta parte para incluir mas rebotes
	Color Le = Color();
	Color fs = Color(1,1,1);
	double factor = 1;
	Color Ld = Color();
	for(int i=0; i<10; i++){
	

	Color fr = spheres[id].c*(1/M_PI); //se debe calcular la fr de acuerdo al material
	
	Vector n = (x-spheres[id].p);
	n.normalize();
	//calcular el color con angulo solido
	Vector wc = spheres[5].p-x;
	double normcx = sqrt(wc.dot(wc));
	wc = wc*(1/normcx);
	double costheta_max = sqrt(1-(spheres[5].r/normcx)*(spheres[5].r/normcx));
	//obtener vector wi
	Vector wi = solidAngle(wc, costheta_max);
	//rayo con origen en x y direccion wi
	Ray shadowRay = Ray(x, wi);
	//verificar la visibilidad con la fuente de luz
	//si el shadow ray intersecta en el mismo id que la fuente de luz, entonces hay visibilidad
	//en caso contrario no hay visibilidad
	double t_aux;
	int id_aux = 0;
	intersect(shadowRay, t_aux, id_aux);
	Point x2 = shadowRay.o + shadowRay.d*t_aux;
	//si hay visibilidad, la contribucion es la radiancia de la fuente de luz
	if(id_aux==5){
		//Le = emision de la fuente de luz 
		Le = spheres[5].radiance*transmitance(x, spheres[5].p, sigma_t);

	//std::cout<<Ld.x<<" "<<Ld.y<<" "<<Ld.z<<std::endl;
	//Ld unicamente es la emision de la fuente de luz con direccion a x
	//integrar con montecarlo de angulo solido
	//fr = albedo
	
	Ld = Le.mult(fr)*(1/solidAngleProb(costheta_max))*n.dot(wi);
	}
	

	//muestrea una dirección en base al material albedo
	Vector wray = cosineHemispheric(n);
	//probabilidad coseno
	double prob = hemiCosineProb(n.dot(wray));
	//rayo con origen en x y direccion wi
	Ray newray = Ray(x, wray);
	Point x_new = Point();
	//ray marching
	Color Lm = rayMarching(newray, sigma_t, sigma_s, segmentos, x_new, id); //se actualiza la id del impacto
	
	//acumular Lm
	Ld=Ld + Lm.mult(fr)*n.dot(wray)*(1/prob);
	//acumular Ld a Lo
	Lo = Lo + Ld.mult(fs)*transmitance(r.o, x, sigma_t)*factor;
	//si Lm es 0, se regresa lo que se acumulo hasta el momento
	if(Lm.x==0 && Lm.y==0 && Lm.z==0) return Lo;
	//guardar la fs en el acumulador
	fs = fs.mult(fr);	 //acumula las fr
	factor = factor*n.dot(wray)*(1/(prob)); //acumula la probabilidad de muestreo
	//actualiza la x
	x = x_new;
	}
		
	//se divide t entre segmentos para obtener la distancia de cada segmento
	double step = t/segmentos;
	//para cada segmento se calcula la transmitancia y la funcion de fase
	for(int i = 0; i<segmentos; i++){
		Point xt = r.o + r.d*step*i;
		//calcular la transmitancia
		double T = transmitance(x, xt, sigma_t);
		//calcular la funcion de fase
		double phase = isotropicPhaseFunction();
		//calcular Ls aproximado
		//lanzar un shadow rays con angulo solido 
		//vector wc de xt a la fuente
		Vector wc = spheres[5].p-xt;
		double normcx = sqrt(wc.dot(wc));
		wc = wc*(1/normcx);
		double costheta_max = sqrt(1-(spheres[5].r/normcx)*(spheres[5].r/normcx));
		//vector wi sampleado con angulo solido
		Vector wi = solidAngle(wc, costheta_max);
		//rayo con origen en xt y direccion wi
		Ray shadowRay = Ray(xt, wi);
		//verificar la visibilidad con la fuente de luz
		//si el shadow ray intersecta en el mismo id que la fuente de luz, entonces hay visibilidad
		//en caso contrario no hay visibilidad
		double t2;
		int id2 = 0;
		intersect(shadowRay, t2, id2);
		//si hay visibilidad, la contribucion es la radiancia de la fuente de luz

		if(id2==5){//se tiene que modificar para que tome en cuenta varias fuentes (workaround)
			//obtener la radiancia de la fuente de luz
			Color Le = spheres[5].radiance;
			//calcular Ls
			//montecarlo de una sola muestra
			Color Ls = Le*(phase*transmitance(xt, spheres[5].p, sigma_t));
			//probabilidad de muestreo
			double prob = solidAngleProb(costheta_max);
			//calcular Ls aproximado
			Color Ls_aprox = Ls*(T*1/prob)*sigma_s*step;
			//acumular Ls aproximado
			Li = Li + Ls_aprox;
		}
		//en caso de no haber visibilidad, la contribucion es 0
		else{
			Color Ls_aprox = Color();
			Li = Li + Ls_aprox;
		}
		


}
	return Li+Lo;
}




//ray marching de paso constante
Color rayMarching2(const Ray &r, double sigma_a, double sigma_s, double step, int idsource){
	
	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;
	Color Li = Color();
	Color Lo = Color();
	//si el rayo intersecta una fuente de luz, Lo = radiancia * transmitancia
	if(spheres[id].radiance.x>0) {
		//return spheres[id].radiance*transmitance(r.o, x, sigma_t);
		Lo = spheres[id].radiance*transmitance(r.o, x, sigma_a+sigma_s);
	}
	//distancia entre x y la fuente de luz divida entre segmento para obtener el nimero de segmentos
	double steps = t/step;
	//ciclo para cada segmento
	for(int i = 0; i<steps; i++){
		Point xt = r.o + r.d*step*i;
		//calcular la transmitancia
		double T = transmitance(x, xt, sigma_a+sigma_s);
		//calcular la funcion de fase
		double phase = isotropicPhaseFunction();
	
		//calcular Ls aproximado
		//lanzar un shadow rays con angulo solido 
		//vector wc de xt a la fuente
		Vector wc = spheres[idsource].p-xt;
		double normcx = sqrt(wc.dot(wc));
		wc = wc*(1/normcx);
		double costheta_max = sqrt(1-(spheres[idsource].r/normcx)*(spheres[idsource].r/normcx));
		//vector wi sampleado con angulo solido
		Vector wi = solidAngle(wc, costheta_max);
		//rayo con origen en xt y direccion wi
		Ray shadowRay = Ray(xt, wi);
		//verificar la visibilidad con la fuente de luz
		//si el shadow ray intersecta en el mismo id que la fuente de luz, entonces hay visibilidad
		//en caso contrario no hay visibilidad
		double t2;
		int id2 = 0;
		intersect(shadowRay, t2, id2);
		//si hay visibilidad, la contribucion es la radiancia de la fuente de luz

		if(id2==idsource){//se tiene que modificar para que tome en cuenta varias fuentes (workaround)
			//obtener la radiancia de la fuente de luz
			Color Le = spheres[idsource].radiance;
			//calcular Ls
			//montecarlo de una sola muestra
			Color Ls = Le*(phase*transmitance(xt, spheres[idsource].p, sigma_a+sigma_s));
			//probabilidad de muestreo
			double prob = solidAngleProb(costheta_max);
			//calcular Ls aproximado
			Color Ls_aprox = Ls*(T*1/prob)*sigma_s*step;
			//acumular Ls aproximado
			Li = Li + Ls_aprox;
		}
		//en caso de no haber visibilidad, la contribucion es 0
		else{
			Color Ls_aprox = Color();
			Li = Li + Ls_aprox;
		}

}
	return Li+Lo;
}

//ray marching de paso constante luz puntual
Color rayMarching3(const Ray &r, double sigma_a, double sigma_s, double step, int idsource){
	
	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;
	Color Li = Color();
	Color Lo = Color();
	/*
	//si el rayo intersecta una fuente de luz, Lo = radiancia * transmitancia
	if(spheres[id].radiance.x>0) {
		//return spheres[id].radiance*transmitance(r.o, x, sigma_t);
		Lo = spheres[id].radiance*transmitance(r.o, x, sigma_a+sigma_s);
	}
	*/
	//distancia entre x y la fuente de luz divida entre segmento para obtener el nimero de segmentos
	double steps = t/step;
	//ciclo para cada segmento
	for(int i = 0; i<steps; i++){
		Point xt = r.o + r.d*step*i;
		//calcular la transmitancia
		double T = transmitance(x, xt, sigma_a+sigma_s);
		//calcular la funcion de fase
		double phase = isotropicPhaseFunction();
		//calcular Ls aproximado
		//lanzar un shadow rays a la fuente puntual 
		//vector wc de xt a la fuente
		Vector wc = spheres[idsource].p-xt;
		//obtener la distancia entre xt y la fuente
		double normwc = wc.dot(wc);
		//verificar la visibilidad con la fuente de luz
		
		if(visibility(spheres[idsource].p,xt)){//se tiene que modificar para que tome en cuenta varias fuentes (workaround)
			//obtener la radiancia de la fuente de luz
			Color Le = spheres[idsource].radiance*(1/normwc);
			//calcular Ls
			//montecarlo de una sola muestra
			Color Ls = Le*(phase*transmitance(xt, spheres[idsource].p, sigma_a+sigma_s));
		
			//calcular Ls aproximado
			Color Ls_aprox = Ls*(T)*sigma_s*step;
			//acumular Ls aproximado
			Li = Li + Ls_aprox;
		}
		//en caso de no haber visibilidad, la contribucion es 0
		else{
			Color Ls_aprox = Color();
			Li = Li + Ls_aprox;
		}

}
	return Li;
}




//muestreo sobre la distancia (para medio homogeneo)
double freeFlightSample(double sigma_t){
	//generar un numero aleatorio
	double xi = erand48(seed);
	//calcular la distancia de vuelo libre
	return -log(1-xi)/sigma_t;
}



//pdf del free flight sampling (para medio homogeneo)
double freeFlightProb(double sigma_t, double d){
	return sigma_t*exp(-sigma_t*d);
}
//pdf success = 1 - pdf failure
double pdfSuccess(double sigma_t, double tmax){
	return 1-exp(-sigma_t*tmax);
}

//pdf failure = transmitance (para medio homogeneo)
double pdfFailure(double sigma_t, double tmax){
	return exp(-sigma_t*tmax);
}



//muestreo de fase isotropica
Vector isotropicPhaseSample(){
	//generar dos numeros aleatorios
	double xi1 = erand48(seed);
	double xi2 = erand48(seed);
	//calcular el angulo theta
	double theta = acos(1-2*xi1);
	//calcular el angulo phi
	double phi = 2*M_PI*xi2;
	//devolver el vector de la direccion del rayo
	Vector wi = Vector(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
	wi.normalize();
	return wi;
}

//pdf de la fase isotropica
double isotropicPhaseProb(){
	return 1/(4*M_PI);
}

//equi-angular sampling
double equiAngularSample(double D, double thetaA, double thetaB){
	double xi = erand48(seed);
	return D*tan((1-xi)*thetaA + xi*thetaB);
}

//pdf equi-angular sampling
double equiAngularProb(double D, double thetaA, double thetaB, double t){
	return D/(thetaB-thetaA)/(t*t+D*D);
}
//volumetric path tracer
Color volumetricPathTracer(const Ray &r, double sigma_a, double sigma_s, int profundidad){
	Color Li = Color();
	Color Lo = Color();
	Color Ls = Color();
	double pdf_binary = 0;	
	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;

	//Transmitancia entre x y r.o
	double T = exp(-(sigma_a+sigma_s)*t); 
	//si intersecta una fuente de luz Lo = radiancia
	if(spheres[id].radiance.x>0) {
		Lo = spheres[id].radiance*T;
		return Lo;

	}
	//si la profundidad llega a 10 se regresa 0
	if(profundidad>=5) return Color();
	//muestrea la distancia de vuelo libre
	double d = freeFlightSample(sigma_a+sigma_s);
	
	//calcular el punto de interseccion xti
	Point x_new = r.o + r.d*d;
	//calcular la transmitancia entre x y xti
	//double T2 = transmitance(x, x_new, sigma_a+sigma_s);
	//calcular la funcion de fase
	double phase = isotropicPhaseFunction();
	//para calcular ls se debe hacer una llamada recursiva a volumetricPathTracer para obtener Li
	//prob = probabilidad del muestreo de fase 
	//obtener la nueva direccion del rayo con muestreo isotropico
	Vector wi_new = isotropicPhaseSample();
	//escoger entre pdf success o pdf failure
	if(d>=t){
		//pdf failure
		pdf_binary = pdfFailure(sigma_a+sigma_s, t);
		return Color();
		
	}
	else{
		//pdf success
		pdf_binary = pdfSuccess(sigma_a+sigma_s, t);
		Li = volumetricPathTracer(Ray(x_new, wi_new), sigma_a, sigma_s, profundidad+1);
	}

	
	double prob = isotropicPhaseProb();
	//Ls = Li*phase*(1/prob); //montecarlo de una sola muestra
	//despejando phase/phase
	Ls = Li;
	//print Li para debug si Li > 0
	
	double sigma_t = sigma_a+sigma_s;

	//montecarlo 
	//Color montecarlo = (Ls*T*sigma_s*(1/freeFlightProb(sigma_a+sigma_s, d)))*(1/pdf_binary);
	Color montecarlo = Ls*(sigma_s/sigma_t); //simplificacion despejando T
	return montecarlo*(1/pdf_binary);	
}

//volumetric path tracer explicito
Color volumetricPathTracerExplicit(const Ray &r, double sigma_a, double sigma_s, int profundidad, int idsource){
	Color Li = Color();
	Color Lo = Color();
	Color Ls = Color();
	Color Ld = Color();
	Point x_new = Point();
	Vector wi_new = Vector();
	double continueprob = 0.9;

	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;
	//Muestreo de fuentes de luz
	//no se debe regresar la radiancia de la fuente de luz dado que ya se muestreo (se puede hacer una excepcion para el primer impacto)
	if(spheres[id].radiance.x>0) {
		if(profundidad==0) return spheres[id].radiance*transmitance(r.o, x, sigma_a+sigma_s);
		else return Color();
	}

	//control de profundidad
	if(profundidad>=5) return Color();
	
	//ruleta rusa
	//if(erand48(seed) < 0.1) return Color();
	
	//muestrea la distancia de vuelo libre, aqui continua como si fuera volumetricPathTracer 
	double d = freeFlightSample(sigma_a+sigma_s);

	double pdf = 0;	
	//escoger entre pdf success o pdf failure
	if(d>=t){
		d = t;
		//pdf failure
		pdf = pdfFailure(sigma_a+sigma_s, t);
		return Color();
		
	}
	else{
		x_new = r.o + r.d*d;
		wi_new = isotropicPhaseSample();
		//pdf success
		pdf = 1-pdfFailure(sigma_a+sigma_s, t);
	}
	

	
	
	//calcular la transmitancia entre x y xti
	double T = transmitance(r.o, x_new, sigma_a+sigma_s);
	//calcular la funcion de fase
	double phase = isotropicPhaseFunction();
	//para calcular ls se debe hacer una llamada recursiva a volumetricPathTracer para obtener Li
	//prob = probabilidad del muestreo de fase
	//obtener la nueva direccion del rayo con muestreo isotropico
	
	//antes de continuar, muestrear la iluminacion directa en x_new
	//calcular el color con angulo solido
	Vector wc = spheres[idsource].p-x_new;
	double normcx = sqrt(wc.dot(wc)); //magnitud de wc
	wc = wc*(1/normcx); //normalizar wc
	double costheta_max = sqrt(1-(spheres[idsource].r/normcx)*(spheres[idsource].r/normcx));
	//vector wi sampleado con angulo solido
	Vector wi_ld = solidAngle(wc, costheta_max);
	//rayo con origen en x_new y direccion wi
	Ray shadowRay = Ray(x_new, wi_ld);
	//verificar la visibilidad con la fuente de luz
	//si el shadow ray intersecta en el mismo id que la fuente de luz, entonces hay visibilidad
	//en caso contrario no hay visibilidad
	double t2;
	int id2 = 0;
	intersect(shadowRay, t2, id2);	
	//si hay visibilidad, la contribucion es la radiancia de la fuente de luz
	if(id2==idsource){
		//obtener la radiancia de la fuente de luz
		Color Le = spheres[idsource].radiance;
		//calcular Ls
		//montecarlo de una sola muestra
		Color Ls = Le*(phase*transmitance(x_new, spheres[idsource].p, sigma_a+sigma_s))*sigma_s;
		//probabilidad de muestreo
		double prob = solidAngleProb(costheta_max);
		//calcular Ls aproximado
		Color Ls_aprox = Ls*T*(1/prob);
		//acumular Ls aproximado
		Ld = Ls_aprox;
	}
	//en caso de no haber visibilidad
	else{
		Color Ls_aprox = Color();
		Ld = Ls_aprox;
	}

	

	Li = volumetricPathTracerExplicit(Ray(x_new, wi_new), sigma_a, sigma_s, profundidad+1, idsource);


	
	double prob = isotropicPhaseProb();
	//Ls = Li*phase*(1/prob); //montecarlo de una sola muestra
	//despejando phase/phase
	Ls = Li;
	
	double sigma_t = sigma_a+sigma_s;
	//montecarlo
	Color montecarlo = Ls*(sigma_s/sigma_t); //simplificacion despejando T, solo es valido con muestreo free flight
	return (Ld*(1/freeFlightProb(sigma_t,d)) + montecarlo)*(1/pdf);

}

//recibe el indice de la fuente de luz y regresa x0, D, thetaA y thetaB
void equiAngularParams(int idsource, Point x, Point &x0, Ray r, double &D, double &thetaA, double &thetaB){
	Point c = spheres[idsource].p;
	//calcular la proyeccion ortogonal de c en el rayo, punto mas cercano a c en el rayo, x0
	x0 = r.o + r.d*((c-r.o).dot(r.d)/(r.d.dot(r.d)));
	//verificar si el punto x0 esta entre r.o y x
	if((x0-r.o).dot(r.d)<0) x0=r.o;
	if((x0-x).dot(r.d)>0){
		x0 = x;
	}

	//calcular la magnitud de x0-c
	D = sqrt((x0-c).dot(x0-c)); 
	//theta A y theta B son los angulos de apertura de la fuente de luz, intervalo de integracion, en este caso usaremos r.o y x como puntos de integracion
	//calcular los lados del triangulo rectangulo, comparten D
	double A = sqrt((x0-r.o).dot(x0-r.o))*-1;
	double B = sqrt((x-x0).dot(x-x0));
	//calcular el angulo theta A y theta B
	thetaA = atan2(A,D);
	thetaB = atan2(B,D);
}

//volumetric path tracer explicito
Color volumetricPathTracerExplicitEquiAngular(const Ray &r, double sigma_a, double sigma_s, int profundidad, int idsource){
	Color Li = Color();
	Color Lo = Color();
	Color Ls = Color();
	Color Ld = Color();
	double continueprob = 0.9;

	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;
	//Muestreo de fuentes de luz
	//no se debe regresar la radiancia de la fuente de luz dado que ya se muestreo (se puede hacer una excepcion para el primer impacto)
	if(spheres[id].radiance.x>0) {
		if(profundidad==0 )return spheres[id].radiance*transmitance(r.o, x, sigma_a+sigma_s);
		else return Color();
	}
	
	//ruleta rusa
	if(erand48(seed) < 0.1) return Color();

	//muestrea la distancia de vuelo libre, aqui continua como si fuera volumetricPathTracer 
	//double d = freeFlightSample(sigma_a+sigma_s);
	
	//calculos necesarios para equi-angular sampling
	double thetaA = 0;
	double thetaB = 0;
	Point c = spheres[idsource].p;
	//calcular la proyeccion ortogonal de c en el rayo, punto mas cercano a c en el rayo, x0
	Point x0 = r.o + r.d*((c-r.o).dot(r.d)/(r.d.dot(r.d)));
	//verificar si el punto x0 esta entre r.o y x
	if((x0-r.o).dot(r.d)<0) x0=r.o;
	if((x0-x).dot(r.d)>0){
		x0 = x;
	}

	//calcular la magnitud de x0-c
	double D = sqrt((x0-c).dot(x0-c)); 
	//theta A y theta B son los angulos de apertura de la fuente de luz, intervalo de integracion, en este caso usaremos r.o y x como puntos de integracion
	//calcular los lados del triangulo rectangulo, comparten D
	double A = sqrt((x0-r.o).dot(x0-r.o))*-1;
	double B = sqrt((x-x0).dot(x-x0));
	//calcular el angulo theta A y theta B

	//if(((t+A)-B)>0.01) std::cout<<t+A-B<<std::endl;

	thetaA = atan2(A,D);
	thetaB = atan2(B,D);
	//muestrear la distancia de con equi-angular sampling
	double d = equiAngularSample(D, thetaA, thetaB);



	//print d para debug
	//std::cout<<A<<"ww"<<d<<"ww"<<B<<"ww"<<t<<std::endl;

	//calcular el punto de interseccion xti,esto en muestreo de free flight
	//Point x_new = r.o + r.d*d;

	Point x_new = x0 + r.d*d;

	
	
	//calcular la transmitancia entre x y xti
	double T = transmitance(r.o, x_new, sigma_a+sigma_s);
	//calcular la funcion de fase
	double phase = isotropicPhaseFunction();
	//para calcular ls se debe hacer una llamada recursiva a volumetricPathTracer para obtener Li
	//prob = probabilidad del muestreo de fase
	//obtener la nueva direccion del rayo con muestreo isotropico
	Vector wi_new = isotropicPhaseSample();
	//antes de continuar, muestrear la iluminacion directa en x_new
	Vector n = (x_new-spheres[id].p);
	n.normalize();
	//calcular el color con angulo solido
	Vector wc = spheres[idsource].p-x_new;
	double normcx = sqrt(wc.dot(wc));
	wc = wc*(1/normcx);
	double costheta_max = sqrt(1-(spheres[idsource].r/normcx)*(spheres[idsource].r/normcx));
	//vector wi sampleado con angulo solido
	Vector wi_ld = solidAngle(wc, costheta_max);
	//rayo con origen en x_new y direccion wi
	Ray shadowRay = Ray(x_new, wi_ld);
	//verificar la visibilidad con la fuente de luz
	//si el shadow ray intersecta en el mismo id que la fuente de luz, entonces hay visibilidad
	//en caso contrario no hay visibilidad
	double t2;
	int id2 = 0;
	intersect(shadowRay, t2, id2);
	//si hay visibilidad, la contribucion es la radiancia de la fuente de luz
	if(id2==idsource){
		//obtener la radiancia de la fuente de luz
		Color Le = spheres[idsource].radiance;
		//calcular Ls
		//montecarlo de una sola muestra
		Color Ls = Le*(phase*transmitance(x_new, spheres[idsource].p, sigma_a+sigma_s))*sigma_s;
		//probabilidad de muestreo
		double prob = solidAngleProb(costheta_max);
		//calcular Ls aproximado
		Color Ls_aprox = Ls*T*(1/prob);
		//acumular Ls aproximado
		Ld = Ls_aprox;
	}
	//en caso de no haber visibilidad
	else{
		Color Ls_aprox = Color();
		Ld = Ls_aprox;
	}

	
	
	Li = volumetricPathTracerExplicitEquiAngular(Ray(x_new, wi_new), sigma_a, sigma_s, profundidad+1, idsource);
	
	double prob = isotropicPhaseProb();
	//Ls = Li*phase*(1/prob); //montecarlo de una sola muestra
	//despejando phase/phase
	Ls = Li;
	
	double sigma_t = sigma_a+sigma_s;
	//montecarlo

	//montercarlo con muestreo equi-angular
	Color montecarlo = (Ls*T*sigma_s);
	return (Ld + montecarlo)*(1/equiAngularProb(D, thetaA, thetaB, d))*(1/continueprob);

}

//volumetric path tracer explicito con equi-angular sampling luz puntual
Color volumetricPathTracerExplicit2(const Ray &r, double sigma_a, double sigma_s, int profundidad, int idsource){
	Color Li = Color();
	Color Lo = Color();
	Color Ls = Color();
	Color Ld = Color();
	Point x_new = Point();
	Vector wi_new = Vector();
	double continueprob = 0.9;
	bool continuar = false;
	double pdf = 0;
	double sigma_t = sigma_a+sigma_s;

	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;


	double q = 1 - continueprob;
	//muestrea la distancia de vuelo libre, aqui continua como si fuera volumetricPathTracer 
	double d = freeFlightSample(sigma_a+sigma_s);
/*	

	//calculos necesarios para equi-angular sampling
	double thetaA = 0;
	double thetaB = 0;
	Point c = spheres[idsource].p;
	//calcular la proyeccion ortogonal de c en el rayo, punto mas cercano a c en el rayo, x0
	Point x0 = r.o + r.d*((c-r.o).dot(r.d)/(r.d.dot(r.d)));
	//verificar si el punto x0 esta entre r.o y x
	if((x0-r.o).dot(r.d)<0) {
		x0 = r.o;
	}
	if((x0-x).dot(r.d)>0){
		x0 = x;
	} 
	


	//calcular la magnitud de x0-c
	double D = sqrt((x0-c).dot(x0-c)); 
	//theta A y theta B son los angulos de apertura de la fuente de luz, intervalo de integracion, en este caso usaremos r.o y x como puntos de integracion
	//calcular los lados del triangulo rectangulo, comparten D
	double A = sqrt((x0-r.o).dot(x0-r.o))*-1;
	double B = sqrt((x-x0).dot(x-x0));
	//calcular el angulo theta A y theta B

	//if(((t+A)-B)>0.01) std::cout<<t+A-B<<std::endl;

	thetaA = atan2(A,D);
	thetaB = atan2(B,D);
	//muestrear la distancia de con equi-angular sampling
	double d = equiAngularSample(D, thetaA, thetaB);
*/

	//print d para debug
	//std::cout<<A<<"ww"<<d<<"ww"<<B<<"ww"<<t<<std::endl;


	
	//escoger entre pdf success o pdf failure
	if(d>=t){
		
		//pdf failure
		pdf = pdfFailure(sigma_t, t);
		d = t;
		return Color();
		
	}
	else{
		//pdf success
		//pdf = sigma_t/(exp(sigma_t*d)-exp(sigma_t*(d-t)));
		pdf = freeFlightProb(sigma_t, d);
		x_new = r.o + r.d*d;
		wi_new = isotropicPhaseSample();
		
		
	}


	

	//x_new = x0 + r.d*d; //para equi-angular sampling

	
	//calcular la transmitancia entre el origen del rayo y el nuevo punto
	double T = transmitance(r.o, x_new, sigma_t);
	//calcular la funcion de fase
	double phase = isotropicPhaseFunction();
	

	
	//verificar visibilidad con luz puntual
	//si hay visibilidad, la contribucion es la radiancia de la fuente de luz
	Point light = spheres[idsource].p;

	if(visibility(light, x_new)){
		//obtener la radiancia de la fuente de luz
		Color Le = spheres[idsource].radiance;
		//calcular ls distancia entre la fuente de luz y x_new
		double distanceLight = (light-x_new).dot(light-x_new);
		//dividir la radiancia por la distancia al cuadrado
		Le = Le*(1/distanceLight);	
		//montecarlo de una sola muestra
		Ls = Le*phase*transmitance(x_new, light, sigma_t);//la probabilidad de muestreo es 1;
		Ld = Ls*T*sigma_s;
	}
	//en caso de no haber visibilidad
	else{
		Ld = Color();
	}

	//if(erand48(seed)<q) return Ld*(1/equiAngularProb(D, thetaA, thetaB, d))*(1/q);
	if(erand48(seed)<q) return Ld*(1/pdf)*(1/q);
	//recursion equi-angular sampling
	//wi_new = isotropicPhaseSample();//obtener la nueva direccion del rayo con muestreo isotropico
	
	
	Li = volumetricPathTracerExplicit2(Ray(x_new, wi_new), sigma_a, sigma_s, profundidad+1, idsource);
	


	//double prob = isotropicPhaseProb();
	//Ls = Li*phase*(1/prob); //multiple scattering
	//despejando phase/phase
	Ls = Li;
	
	//montecarlo
	//Color montecarlo = Ls*(sigma_s/sigma_t); //simplificacion despejando T, solo es valido con muestreo free flight
	//montercarlo con muestreo equi-angular
	Color montecarlo = (Ls*T*sigma_s);
	//return ((Ld + montecarlo)*(1/equiAngularProb(D, thetaA, thetaB, d)))*(1/continueprob);//equiangular sampling 

	return (Ld*(1/pdf) + montecarlo*(1/pdf))*(1/continueprob);

}

//volumetric path tracer explicito VPT para multiples fuentes de luz
Color volumetricPathTracer3(const Ray &r, double sigma_a, double sigma_s, int profundidad){
	Color Li = Color();
	Color Lo = Color();
	Color Ls = Color();
	Color Ld = Color();
	Point x_new = Point();
	Vector wi_new = Vector();
	double continueprob = 0.5;
	double pdf = 0;
	double sigma_t = sigma_a+sigma_s;

	double phase = isotropicPhaseFunction();
	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;

	double q = 1 - continueprob;
	//escoger una fuente de luz para conexion directa
	int idsource = 0;
	
	//arreglo para las fuentes (por el momemnto estatico)
	int arr[4] = {-1,-1,-1,-1};

	//sizeof spheres
	int n = sizeof(spheres)/sizeof(spheres[0]);
	int j = 0;
	//guardar en las posiciones del arreglo las fuentes de luz
	for(int i = 0; i<n; i++){
		if(spheres[i].radiance.x>0 || spheres[i].radiance.y>0 || spheres[i].radiance.z>0){
			arr[j] = i;
			j++;
		} 
	}

	//contar las fuentes

	int count = 0;
	for(int i = 0; i<4; i++){
		if(arr[i]!=-1) count++;
	}
	
	if(count==0) return Color();
	//para metodo 2 muestrear una fuente de luz 

	//probabildad uniforme para cada fuente
	double prob = 1.0/count;

	

	
	//muestrear una fuente de luz
	idsource = arr[(int)(erand48(seed)*count)];

/* //en caso de hacer muestreo equi-angular
	double D = 0;
	double thetaA = 0;
	double thetaB = 0;
	Point x0 = Point();
	equiAngularParams(idsource, x, x0, r, D, thetaA, thetaB);
	double d = equiAngularSample(D, thetaA, thetaB);
*/

	//muestrea la distancia de vuelo libre, aqui continua como si fuera volumetricPathTracer
	double d = freeFlightSample(sigma_a+sigma_s);
	
	//escoger entre pdf success o pdf failure
	if(d>=t){
		
		//pdf failure
		pdf = pdfFailure(sigma_t, t);
		d = t;
		return Color();
		
	}
	else{
		//pdf success
		//pdf = sigma_t/(exp(sigma_t*d)-exp(sigma_t*(d-t)));
		pdf = freeFlightProb(sigma_t, d);
		x_new = r.o + r.d*d;
		wi_new = isotropicPhaseSample();
		
		
	}
	
/*
	x_new = x0 + r.d*d; //para equi-angular sampling
	pdf = equiAngularProb(D, thetaA, thetaB, d);
	wi_new = isotropicPhaseSample();
*/
	//calcular la transmitancia entre el origen del rayo y el nuevo punto
	double T = transmitance(r.o, x_new, sigma_t);

	//calcula la contribucion directa de la fuente de luz
	//determinar si es puntual o esferica
	if(spheres[idsource].r==0){
		//ejecutar rutina de luz puntual
		Point light = spheres[idsource].p;
		if(visibility(light, x_new)){
		//obtener la radiancia de la fuente de luz
		Color Le = spheres[idsource].radiance;
		//calcular ls distancia entre la fuente de luz y x_new
		double distanceLight = (light-x_new).dot(light-x_new);
		//dividir la radiancia por la distancia al cuadrado
		Le = Le*(1/distanceLight);	
		//montecarlo de una sola muestra
		Ls = Le*phase*transmitance(x_new, light, sigma_t);//la probabilidad de muestreo es 1;
		Ld = Ls*T*sigma_s*(1/prob); //1/prob por el evento de muestreo de la fuente
		}
		//en caso de no haber visibilidad
		else	Ld = Color();
	}
	else{
		//ejecutar rutina de angulo solido
	}

	if(erand48(seed)<q) return Ld*(1/pdf)*(1/q);
	//recursion
	Li = volumetricPathTracer3(Ray(x_new, wi_new), sigma_a, sigma_s, profundidad+1);

	//probabilidad de muestreo de fase se cancela con phase/phase
	Ls = Li;
	
	//montecarlo
	Color montecarlo = (Ls*T*sigma_s);
	return (Ld*(1/pdf) + montecarlo*(1/pdf))*(1/continueprob);

}

//volumetric path tracer explicito VPT para multiples fuentes de luz, version alternativa 
Color volumetricPathTracer3alt(const Ray &r, double sigma_a, double sigma_s, int profundidad){
	Color Li = Color();
	Color Lo = Color();
	Color Ls = Color();
	Color Ld = Color();
	Point x_new = Point();
	Vector wi_new = Vector();
	double continueprob = 0.5;
	double pdf = 0;
	double sigma_t = sigma_a+sigma_s;

	double phase = isotropicPhaseFunction();
	//obtener la distancia de interseccion
	double t;
	int id = 0;
	if (!intersect(r, t, id))
		return Color();	// el rayo no intersecto objeto, return Vector() ==
	Point x = r.o + r.d*t;

	double q = 1 - continueprob;
	//escoger una fuente de luz para conexion directa
	int idsource = 0;
	
	//arreglo para las fuentes (por el momemnto estatico)
	int arr[4] = {-1,-1,-1,-1};

	//sizeof spheres
	int n = sizeof(spheres)/sizeof(spheres[0]);
	int j = 0;
	//guardar en las posiciones del arreglo las fuentes de luz
	for(int i = 0; i<n; i++){
		if(spheres[i].radiance.x>0 || spheres[i].radiance.y>0 || spheres[i].radiance.z>0){
			arr[j] = i;
			j++;
		} 
	}

	//contar las fuentes

	int count = 0;
	for(int i = 0; i<4; i++){
		if(arr[i]!=-1) count++;
	}
	
	if(count==0) return Color();
	//solo se puede hacer free flight sampling
	double d = freeFlightSample(sigma_t);

	

	if(d>=t){
		
			//pdf failure
			pdf = pdfFailure(sigma_t, t);
			d = t;
			return Color();
		
		}
	else{
			//pdf success
			//pdf = sigma_t/(exp(sigma_t*d)-exp(sigma_t*(d-t)));
			pdf = freeFlightProb(sigma_t, d);
			x_new = r.o + r.d*d;
			wi_new = isotropicPhaseSample();
		}
	
	//calcular la transmitancia entre el origen del rayo y el nuevo punto
	double T = transmitance(r.o, x_new, sigma_t);
	
	//para metodo 1 iterar sobre arr
	Color accum = Color();
	for(int i = 0; i<count; i++){
		
		//para cada indice de fuente de luz
		idsource = arr[i];

		//calcula la contribucion directa de la fuente de luz
		//determinar si es puntual o esferica
		if(spheres[idsource].r==0){
			//ejecutar rutina de luz puntual
			Point light = spheres[idsource].p;
			if(visibility(light, x_new)){
			//obtener la radiancia de la fuente de luz
			Color Le = spheres[idsource].radiance;
			//calcular ls distancia entre la fuente de luz y x_new
			double distanceLight = (light-x_new).dot(light-x_new);
			//dividir la radiancia por la distancia al cuadrado
			Le = Le*(1/distanceLight);	
			//montecarlo de una sola muestra
			Ls = Le*phase*transmitance(x_new, light, sigma_t);//la probabilidad de muestreo es 1;
			Ld = Ls*T*sigma_s; //1/prob por el evento de muestreo de la fuente
			}
			//en caso de no haber visibilidad
			else	Ld = Color();
		}
		else{
			//ejecutar rutina de angulo solido
			}
		accum = accum + Ld;

	}

	if(erand48(seed)<q) return accum*(1/pdf)*(1/q);
	//recursion
	Li = volumetricPathTracer3alt(Ray(x_new, wi_new), sigma_a, sigma_s, profundidad+1);

	//probabilidad de muestreo de fase se cancela con phase/phase
	Ls = Li;
	
	//montecarlo
	Color montecarlo = (Ls*T*sigma_s);
	return (accum*(1/pdf) + montecarlo*(1/pdf))*(1/continueprob);

}

//luz puntual single scattering v1 para ray marching
Color punctualVolumetric(int idsource, Point x, double phase, double sigma_t, double sigma_s){
	Color Ls = Color();
	Color Ld = Color();
	Point light = spheres[idsource].p;
	if(visibilityVPT(light, x)){
		//obtener la radiancia de la fuente de luz
		Color Le = spheres[idsource].radiance;
		//calcular ls distancia entre la fuente de luz y x_new
		double distanceLight = (light-x).dot(light-x);
		//dividir la radiancia por la distancia al cuadrado
		Le = Le*(1/distanceLight);	
		//montecarlo de una sola muestra
		Ls = Le*phase*multipleT(x, light, sigma_t);//la probabilidad de muestreo es 1;
		Ld = Ls*sigma_s;
	}
	//en caso de no haber visibilidad
	else	Ld = Color();
	return Ld;

}

//funcion para calcular la suma de las transmitacias entre dos puntos
//se debe considerar que pueden existir objetos volumetricos en el camino y espacios vacios
//por ahora todos tienen la misma sigma_t, pero se debe tomar de la interseccion
double multipleT(Point x1, Point x2 , double sigma_t){
	//las transmitancias se multiplican
	double T = 1;
	double t_cercano;
	int id = 0;
	double t_aux, t_aux2;
	//lanzar un rayo entre x1 y x2
	Vector w = x2-x1;
	w.normalize();
	Ray r = Ray(x1, w);
	//calcula la interseccion con cada esfera volumetrica, si no hay devuelve 1
	//for each volumetric object
	for(int i = 0; i<sizeof(spheres)/sizeof(spheres[0]); i++){
		//si es tipo 3
		if(spheres[i].material==3){
			spheres[i].intersectVPT(r,t_aux,t_aux2 ); //tiene que ser una version de intersect que regrese t1 y t2
			if(t_aux2<0){
				T = T*exp(-sigma_t*t_aux);
			}
			//se debe cumplir que t1-t2>0
			if(t_aux2-t_aux>0){
				//calcular la transmitancia entre x1 y x2
				T = T*exp(-sigma_t*(t_aux2-t_aux));
			}

		}

		}
	
	return T;
}

//path tracing explicito recursivo con MIS y ruleta rusa constante, se incluyen objetos volumetricos
inline Color explicitPathRecursive2(const Ray &r, int bounce){
	double sigma_a = 0.05;
	double sigma_s = 0.009;
	double sigma_t = sigma_a+sigma_s;
	double t,t2;
	int id = 0;
	Color Ls = Color();


	Color  fs, Ld, Lind;

	//obtener la cantidad de esferas en la escena
	int a = sizeof(spheres);
	int b = sizeof(spheres[0]);
	int size = a/b; //cantidad de esferas en la escena

	if (!intersectV2(r, t,t2, id))
		return Color();	// el rayo no intersecto objeto, return Vector() == negro
  

	if(spheres[id].radiance.x > 0)
		return Color();

	Point x = r.o + r.d*t;
	
	Point xt;

	if(spheres[id].material==3){
		//haz ray marching hasta terminar la esfera
		int steps = 100;
		//para calcular el paso se debe conocer la distancia al limite de la esfera
		double distance = t2-t;
		double step = distance/steps;
		for (int i=0; i<steps; i++){
			xt = x + r.d*step*i;
			//for each light source
			for(int light = 0; light<size; light++){
				//resuelve las fuentes puntuales
				if(spheres[light].r==0){
					Ls = punctualVolumetric(light,xt,isotropicPhaseFunction(), sigma_t, sigma_s)*step*transmitance(x,xt,sigma_t)+Ls;
				}
			}
	

		}
		return Ls + explicitPathRecursive2(Ray(xt,r.d),bounce)*transmitance(x,xt, sigma_t);

	}
	


	Vector n = (x-spheres[id].p);  
	n.normalize();
	
	
	Vector wi, wh;
	Vector wo = r.d * -1;
	
	double cosine, prob;
	
	//calcular Ld for each light 
	//for each light source

	for(int light = 0; light<size; light++) {
			//resuelve las fuentes puntuales
			if(spheres[light].r==0) Ld = pLight(spheres[id], x, n, r.d, spheres[light].radiance, spheres[light].p, spheres[id].alpha) + Ld;
	}
	Ld = MIS(spheres[id], x, n, r.d, spheres[id].alpha) + Ld;

	//ruleta rusa constante
	
	double q = 0.1;
	double continueprob = 1.0 - q;
	if(erand48(seed) < q)
		return Ld; // no continuar el camino (regresa 0 en la integral)
	//muestreo de BDSF para obtener Lind
	if(spheres[id].material==0){
		wi = cosineHemispheric(n);
		fs = spheres[id].c*(1/M_PI);
		prob = hemiCosineProb(n.dot(wi));
	}
	else{
		double alpha = 0.001;
		Vector wh = vectorFacet(alpha); //local
		Vector s, t;
		coordinateSystem(n, s, t);
		wh = s*wh.x+t*wh.y+n*wh.z; //global
		wi = wo*(-1)+wh*2*(wh.dot(wo));
		fs = frMicroFacet(spheres[id].eta, spheres[id].kappa, wi, wh, wo, alpha, n);
		prob = microFacetProb(wo, wh, alpha, n);
	}

	
	Ray recursiveRay = Ray(x, wi);
	cosine = n.dot(wi);
	bounce++;
	Lind = fs.mult(explicitPathRecursive2(recursiveRay, bounce))*abs(cosine)*(1/(prob*continueprob));

	return (Ld + Lind);	
	//return explicitPathRecursive(recursiveRay, bounce, Ld, fs, cosine, prob)
}

	


int main(int argc, char *argv[]) {
	// semilla para generador de números aleatorios
	while(getentropy(seed,3));

	std::chrono::time_point<std::chrono::system_clock> start, end;

	start = std::chrono::system_clock::now();

	int w = 1024, h = 768; // image resolution
  
	// fija la posicion de la camara y la dirección en que mira
	Ray camera( Point(0, 11.2, 214), Vector(0, -0.042612, -1).normalize() );

	// parametros de la camara
	Vector cx = Vector( w * 0.5095 / h, 0., 0.); 
	Vector cy = (cx % camera.d).normalize() * 0.5095;
  
	// auxiliar para valor de pixel y matriz para almacenar la imagen
	Color *pixelColors = new Color[w * h];


	// PROYECTO 1
	// usar openmp para paralelizar el ciclo: cada hilo computara un renglon (ciclo interior),
	#pragma omp parallel for schedule(dynamic, 1)
	for(int y = 0; y < h; y++) 
	{ 	
		// recorre todos los pixeles de la imagen
		fprintf(stderr,"\r%5.2f%%",100.*y/(h-1));
		for(int x = 0; x < w; x++ ) {
			int idx = (h - y - 1) * w + x; // index en 1D para una imagen 2D x,y son invertidos
			Color pixelValue = Color(); // pixelValue en negro por ahora
			
			// para el pixel actual, computar la dirección que un rayo debe tener
			//Vector cameraRayDir = cx * (double(x)/w - .5) + cy * (double(y)/h - .5) + camera.d;
			
			// computar el color del pixel para el punto que intersect*o el rayo desde la camara
			
			//pixelValue = pathTracer( Ray(camera.o, cameraRayDir.normalize()));
		
			//montercarlo path tracing con n rpp (rays per pixel)
			int rpp = atoi(argv[1]); //en esta implementacion tenemos 1 muestra por rayo desde la camara 1 rpp = 1 spp

			for(int i=0; i<rpp; i++){
				Vector cameraRayDir = cx * ((double(x)+erand48(seed)-0.5)/w - .5) + cy * ((double(y)+erand48(seed)-0.5)/h - .5) + camera.d;
			
				// computar el color del pixel para el punto que intersectó el rayo desde la camara

				pixelValue = explicitPathRecursive2(Ray(camera.o, cameraRayDir.normalize()),0)+ pixelValue;
				//pixelValue = iterativePathTracer(Ray(camera.o, cameraRayDir.normalize()))+ pixelValue;
				//pixelValue = tailExplicitPath(Ray(camera.o, cameraRayDir.normalize()), 0, Color(0,0,0),Vector(1,1,1), 1)+pixelValue;
				//pixelValue = rayMarching2(Ray(camera.o, cameraRayDir.normalize()), 0.001,0.009, 0.1, 7)+pixelValue;

				//pixelValue = rayMarchingGlobal(Ray(camera.o, cameraRayDir.normalize()),0.001,0.0125,10)+pixelValue;
				//pixelValue = volumetricPathTracer(Ray(camera.o, cameraRayDir.normalize()), 0.001, 0.009, 0)+pixelValue;
				//pixelValue = volumetricPathTracerExplicit2(Ray(camera.o, cameraRayDir.normalize()), 0.001, 0.009, 0,5)+pixelValue;
				//pixelValue = volumetricPathTracerExplicitEquiAngular(Ray(camera.o, cameraRayDir.normalize()), 0.001, 0.009, 0,7)+pixelValue;
				//pixelValue = volumetricPathTracer3(Ray(camera.o, cameraRayDir.normalize()), 0.001, 0.009, 0)+pixelValue;
			}

			pixelValue = pixelValue * (1/(double)rpp); //promedio de color de cada pixel

			// limitar los tres valores de color del pixel a [0,1]
			pixelColors[idx] = Color(clamp(pixelValue.x), clamp(pixelValue.y), clamp(pixelValue.z));
		}
	}
	

	fprintf(stderr,"\n");

	// PROYECTO 1
	// Investigar formato ppm
	FILE *f = fopen("image.ppm", "w");
	// escribe cabecera del archivo ppm, ancho, alto y valor maximo de color
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
	for (int p = 0; p < w * h; p++) 
	{ // escribe todos los valores de los pixeles
    		fprintf(f,"%d %d %d ", toDisplayValue(pixelColors[p].x), toDisplayValue(pixelColors[p].y), 
				toDisplayValue(pixelColors[p].z));
  	}
  	fclose(f);

  	delete[] pixelColors;

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout<< "elapsed time: " << elapsed_seconds.count() << "s\n";

	return 0;
}
