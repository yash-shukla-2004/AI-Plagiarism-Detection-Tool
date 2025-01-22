import javalang
import re


java_code = """
// This is a simple Java program
public class HelloWorld {
    public static void main(String[] args) {
        int a = 5;
        int b = 26;
        System.out.println("Hello, World!"); // Print greeting
        System.out.println(a+b); // Print sum
    }
}
"""

def tokenize(code):

    tokens = []
    try:
        for token in javalang.tokenizer.tokenize(code):
            tokens.append(token.value)

    except javalang.tokenizer.LexerError:
        print("Error tokenizing the Java Code")
    return tokens




def remove_comments(code):
    code = re.sub(r"//.*","", code)
    code = re.sub(r"/\*.*?\*/","",code,flags=re.DOTALL)
    return code




def normalize(code):
    return re.sub(r"\s+"," ",code).strip()


def tokenize_using_lexical_analysis(java_code):
    """
    Tokenizes Java code using lexical analysis and abstracts syntax variations.
    """
    tokens = []
    try:
        # Tokenize using javalang
        for token in javalang.tokenizer.tokenize(java_code):
            if isinstance(token, javalang.tokenizer.Keyword):
                tokens.append(token.value)  # Keep keywords
            elif isinstance(token, javalang.tokenizer.Identifier):
                tokens.append("IDENTIFIER")  # Abstract identifiers
            elif token.value.isdigit():
                tokens.append("LITERAL")  # Abstract literals
            elif isinstance(token, javalang.tokenizer.String):
                tokens.append("STRING_LITERAL")  # Abstract string literals
            else:
                tokens.append(token.value)  # Keep symbols/operators
    except javalang.tokenizer.LexerError:
        print("Error tokenizing Java code.")
    return tokens


def tokenize_using_ast(java_code):
    """
    Tokenizes Java code based on its Abstract Syntax Tree (AST).
    """
    tokens = []
    try:
        # Parse the code into an AST
        tree = javalang.parse.parse(java_code)
        
        # Traverse the AST and extract structural elements
        for path, node in tree:
            # Append the type of node (e.g., MethodDeclaration, IfStatement)
            tokens.append(type(node).__name__)
            
            # Optionally, include method/loop names but ignore variable details
            if isinstance(node, javalang.tree.MethodDeclaration):
                tokens.append("METHOD")
            elif isinstance(node, javalang.tree.IfStatement):
                tokens.append("IF")
            elif isinstance(node, javalang.tree.ForStatement):
                tokens.append("FOR")
            elif isinstance(node, javalang.tree.WhileStatement):
                tokens.append("WHILE")
    except javalang.parser.JavaSyntaxError:
        print("Error parsing Java code.")
    return tokens


def abstract_identifiers(tokens):
    """
    Replaces user-defined identifiers with generic placeholders like var_1, var_2.
    """
    identifier_map = {}
    abstracted_tokens = []
    identifier_count = 1

    # Define a set of Java keywords and built-in terms to ignore
    java_keywords = {
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
        "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float",
        "for", "if", "goto", "implements", "import", "instanceof", "int", "interface", "long", "native",
        "new", "null", "package", "private", "protected", "public", "return", "short", "static",
        "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try",
        "void", "volatile", "while", "true", "false", "String", "out", "println"
    }
    built_in_classes = {"System", "Math", "Integer", "Double", "Character"}  # Extend as needed

    for token in tokens:
        # Check if the token is a user-defined identifier
        if token.isidentifier() and token not in java_keywords and token not in built_in_classes:
            if token not in identifier_map:
                identifier_map[token] = f"var_{identifier_count}"
                identifier_count += 1
            abstracted_tokens.append(identifier_map[token])
        else:
            # Preserve keywords, built-in classes, and literals
            abstracted_tokens.append(token)

    return abstracted_tokens


def preprocessing(code):

    code = remove_comments(code)

    code = normalize(code)

    tokens = tokenize(code)

    abstracted_tokens = abstract_identifiers(tokens)

    tokens_lex = tokenize_using_lexical_analysis(code)

    tokens_ast = tokenize_using_ast(code)

    
    print(f"tokens: {tokens}")
    print(f"abstracted tokens: {abstracted_tokens}")
    print(f"tokens using ast : {tokens_ast}")
    print(f"tokens using lexical analysis: {tokens_lex}")
    return {"tokens":tokens ,"abstracted tokens": abstracted_tokens, "tokens_lex":tokens_lex,"tokens_ast":tokens_ast, "code" : code}


preprocessing(java_code)