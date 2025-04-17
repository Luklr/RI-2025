class DynamicMatrix:
    def __init__(self):
        self.data = [[""]]  # Matriz inicial con celda [0,0]
        self.row_keys = {}   # {"term": row_index}
        self.col_keys = {}   # {"doc_name": col_index}
        self.next_row = 1    # Siguiente fila disponible
        self.next_col = 1    # Siguiente columna disponible

    def __getitem__(self, indexes):
        row, col = indexes
        return self.data[row][col]

    def __setitem__(self, indexes, value):
        row, col = indexes
        # Asegurar que la matriz sea lo suficientemente grande
        while row >= len(self.data):
            self.data.append([0] * len(self.data[0]))
        while col >= len(self.data[row]):
            self.data[row].append(0)
        self.data[row][col] = value

    def add_row(self, key=None):
        """Agrega una nueva fila, opcionalmente con una clave (debe ser string o None)."""
        row_index = self.next_row
        self.next_row += 1
        
        # Asegurar que todas las filas tengan el mismo número de columnas
        new_row = [0] * len(self.data[0])
        self.data.append(new_row)
        
        if key is not None:
            if isinstance(key, list):
                # Si es una lista, usamos el primer elemento como clave
                key = str(key[0]) if len(key) > 0 else ""
            self.row_keys[str(key)] = row_index
            self[row_index, 0] = key  # Guardar key en primera columna
            
        return row_index

    def add_column(self, key=None):
        """Agrega una nueva columna, opcionalmente con una clave."""
        col_index = self.next_col
        self.next_col += 1
        
        for row in self.data:
            row.append(0)  # Añadir celda a cada fila
            
        if key is not None:
            self.col_keys[key] = col_index
            self[0, col_index] = key  # Guardar key en primera fila
            
        return col_index

    def get_row_index(self, key):
        """Obtiene el índice de fila por clave, o -1 si no existe."""
        return self.row_keys.get(key, -1)

    def get_col_index(self, key):
        """Obtiene el índice de columna por clave, o -1 si no existe."""
        return self.col_keys.get(key, -1)

    def ensure_term_exists(self, term):
        """Garantiza que un término existe, devuelve su índice de fila."""
        row_index = self.get_row_index(term)
        if row_index == -1:
            row_index = self.add_row(term)
        return row_index

    def ensure_doc_exists(self, doc_name):
        """Garantiza que un documento existe, devuelve su índice de columna."""
        col_index = self.get_col_index(doc_name)
        if col_index == -1:
            col_index = self.add_column(doc_name)
        return col_index

    def columns(self):
        return len(self.data[0]) if self.data else 0

    def rows(self):
        return len(self.data)

    def print_rows(self):
        for row in self.data:
            print(row)