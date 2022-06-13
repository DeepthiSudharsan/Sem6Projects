// package mcd_project;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class html2Text {

	public static String htm2Text(String html) {
		StringBuilder sb = new StringBuilder();
		int state = 0;

		int styleStart = html.toLowerCase().indexOf("<style");

		int styleEnd = html.toLowerCase().indexOf("/style>");

		while (styleStart != -1 && styleEnd > styleStart) {
			html = html.substring(0, styleStart) + html.substring(styleEnd + "/style>".length());
			styleStart = html.toLowerCase().indexOf("<style");
			styleEnd = html.toLowerCase().indexOf("/style>");
		}

		int commentStart = html.indexOf("<!--");
		int commentEnd = html.indexOf("-->");
		while (commentStart != -1 && commentEnd > commentStart) {
			html = html.substring(0, commentStart) + html.substring(commentEnd + "-->".length());
			commentStart = html.indexOf("<!--");
			commentEnd = html.indexOf("-->");
		}

		html = html.trim();// removing blank spaces at the end

		for (char ch : html.toCharArray()) {
			switch (state) {
			case (int) '<':
				// Read to null until '>'-char is read
				if (ch != '>')
					continue;
				state = 0;
				break;
			default:
				if (ch == '<') {
					state = '<';
					continue;
				}
				sb.append(ch);
				break;
			}
		}
		return sb.toString().trim();

//	    System.out.print(s);
	}

	public static String read1(String filename) throws FileNotFoundException {
		File file = new File(filename);
		@SuppressWarnings("resource")
		Scanner sc = new Scanner(file);
		String fin_string = "";
		while (sc.hasNextLine())
			// System.out.println(sc.nextLine());
			fin_string += sc.nextLine() + "\n";
		return fin_string.trim();
	}

	public static void main(String[] args) throws FileNotFoundException {
		String s = htm2Text("<p>Hello string</p>");
		// System.out.print(s);
		String x = read1("html.txt");
		String s1 = htm2Text(x);
		System.out.println(s1);

	}
}

//Above code in python
//import re
//import sys
//class html2Text:
//    def __init__(self):
//        self.state = 0
//        self.s = ""
//    def htm2Text(self, html):
//        self.s = ""
//        for ch in html:
//            if self.state == '<':
//                if ch != '>':
//                    continue
//                self.state = 0
//            else:
//                if ch == '<':
//                    self.state = '<'
//                    continue
//                self.s += ch
//        return self.s.strip()
//    def read1(self, filename):
//        f = open(filename, 'r')
//        fin_string = ""
//        for line in f:
//            fin_string += line
//        return fin_string.strip()
//    def main(self):
//        s = self.htm2Text("<p>Hello string</p>")
//        print s
//        s1 = self.read1("/Users/roshantushar/Downloads/html.txt")
//        s2 = self.htm2Text(s1)
//        print s2
//
//if __name__ == "__main__":
//    htm2Text().main()
//
//
//